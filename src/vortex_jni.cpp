#include "gpu_vector_index.cuh"

#include <jni.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef VORTEX_NATIVE_VERSION
#define VORTEX_NATIVE_VERSION "unknown"
#endif

#ifndef VORTEX_GIT_COMMIT
#define VORTEX_GIT_COMMIT "unknown"
#endif

#ifndef VORTEX_GIT_DIRTY
#define VORTEX_GIT_DIRTY 1
#endif

namespace
{

constexpr jint PRECISION_FP32_EXACT = 0;
constexpr jint PRECISION_FP16_STORAGE_FP32_ACCUMULATION = 1;

struct NativeIndex
{
    explicit NativeIndex(
        const std::vector<float>& database,
        const int vectorCount,
        const int dimension)
        : index(
            database,
            vectorCount,
            dimension
        )
    {
    }

    // GpuVectorIndex deliberately reuses persistent query/candidate
    // buffers for steady-state performance. Serialize calls against one
    // index instance so Java callers cannot race those buffers.
    std::mutex searchMutex;

    GpuVectorIndex index;
};

std::mutex registryMutex;

std::unordered_map<
    jlong,
    std::shared_ptr<NativeIndex>
> registry;

std::atomic<jlong> nextHandle{
    1
};

void throwJavaException(
    JNIEnv* environment,
    const char* className,
    const std::string& message)
{
    jclass exceptionClass =
        environment->FindClass(
            className
        );

    if (exceptionClass == nullptr)
    {
        return;
    }

    environment->ThrowNew(
        exceptionClass,
        message.c_str()
    );

    environment->DeleteLocalRef(
        exceptionClass
    );
}

void throwIllegalArgument(
    JNIEnv* environment,
    const std::string& message)
{
    throwJavaException(
        environment,
        "java/lang/IllegalArgumentException",
        message
    );
}

void throwIllegalState(
    JNIEnv* environment,
    const std::string& message)
{
    throwJavaException(
        environment,
        "java/lang/IllegalStateException",
        message
    );
}

std::size_t checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    const char* message)
{
    if (left != 0 &&
        right >
            static_cast<std::size_t>(
                -1
            ) /
            left)
    {
        throw std::overflow_error(
            message
        );
    }

    return
        left *
        right;
}

std::shared_ptr<NativeIndex>
requireIndex(
    const jlong handle)
{
    if (handle <= 0)
    {
        throw std::invalid_argument(
            "Native index handle must be positive."
        );
    }

    std::lock_guard<std::mutex>
        lock(
            registryMutex
        );

    const auto iterator =
        registry.find(
            handle
        );

    if (iterator ==
        registry.end())
    {
        throw std::runtime_error(
            "Native index handle is closed or invalid."
        );
    }

    return
        iterator->second;
}

std::vector<float> copyFloatArray(
    JNIEnv* environment,
    jfloatArray source,
    const std::size_t expectedLength,
    const char* argumentName)
{
    if (source == nullptr)
    {
        throw std::invalid_argument(
            std::string(argumentName) +
            " must not be null."
        );
    }

    const jsize actualLength =
        environment->GetArrayLength(
            source
        );

    if (actualLength < 0 ||
        static_cast<std::size_t>(
            actualLength
        ) !=
            expectedLength)
    {
        throw std::invalid_argument(
            std::string(argumentName) +
            " has an unexpected length."
        );
    }

    std::vector<float> values(
        expectedLength
    );

    if (!values.empty())
    {
        environment->GetFloatArrayRegion(
            source,
            0,
            actualLength,
            values.data()
        );

        if (environment->ExceptionCheck())
        {
            throw std::runtime_error(
                "Unable to copy float array from Java."
            );
        }
    }

    return values;
}

void validateOutputArray(
    JNIEnv* environment,
    const jintArray output,
    const std::size_t expectedLength,
    const char* argumentName)
{
    if (output == nullptr)
    {
        throw std::invalid_argument(
            std::string(argumentName) +
            " must not be null."
        );
    }

    const jsize actualLength =
        environment->GetArrayLength(
            output
        );

    if (actualLength < 0 ||
        static_cast<std::size_t>(
            actualLength
        ) !=
            expectedLength)
    {
        throw std::invalid_argument(
            std::string(argumentName) +
            " has an unexpected length."
        );
    }
}

void validateOutputArray(
    JNIEnv* environment,
    const jfloatArray output,
    const std::size_t expectedLength,
    const char* argumentName)
{
    if (output == nullptr)
    {
        throw std::invalid_argument(
            std::string(argumentName) +
            " must not be null."
        );
    }

    const jsize actualLength =
        environment->GetArrayLength(
            output
        );

    if (actualLength < 0 ||
        static_cast<std::size_t>(
            actualLength
        ) !=
            expectedLength)
    {
        throw std::invalid_argument(
            std::string(argumentName) +
            " has an unexpected length."
        );
    }
}

GpuBatchTopKResult executeSearch(
    NativeIndex& nativeIndex,
    const std::vector<float>& queries,
    const int queryCount,
    const int topK,
    const jint precision)
{
    if (precision ==
        PRECISION_FP32_EXACT)
    {
        return
            nativeIndex.index
                .searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    queryCount,
                    topK
                );
    }

    if (precision ==
        PRECISION_FP16_STORAGE_FP32_ACCUMULATION)
    {
        return
            nativeIndex.index
                .searchTopKBatchTiledFp16Auto(
                    queries,
                    queryCount,
                    topK
                );
    }

    throw std::invalid_argument(
        "Unsupported search precision code."
    );
}

template <typename Function>
auto translateExceptions(
    JNIEnv* environment,
    Function&& function)
    -> decltype(function())
{
    using ReturnType =
        decltype(
            function()
        );

    try
    {
        return
            function();
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );
    }
    catch (...)
    {
        throwIllegalState(
            environment,
            "Unknown native Vortex CUDA failure."
        );
    }

    if constexpr (
        std::is_pointer_v<
            ReturnType
        >
    )
    {
        return nullptr;
    }
    else
    {
        return ReturnType{};
    }
}

template <typename Function>
void translateVoidExceptions(
    JNIEnv* environment,
    Function&& function)
{
    try
    {
        function();
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );
    }
    catch (...)
    {
        throwIllegalState(
            environment,
            "Unknown native Vortex CUDA failure."
        );
    }
}

} // namespace

extern "C"
{

JNIEXPORT jstring JNICALL
Java_com_vortex_cuda_VortexNative_nativeVersion(
    JNIEnv* environment,
    jclass)
{
    std::ostringstream stream;

    stream
        << "Vortex CUDA JNI "
        << VORTEX_NATIVE_VERSION
        << " (commit "
        << VORTEX_GIT_COMMIT
        << ", "
        << (
            VORTEX_GIT_DIRTY
                ? "dirty"
                : "clean"
        )
        << ")";

    return
        environment->NewStringUTF(
            stream.str().c_str()
        );
}

JNIEXPORT jlong JNICALL
Java_com_vortex_cuda_VortexNative_nativeCreateIndex(
    JNIEnv* environment,
    jclass,
    jfloatArray databaseArray,
    const jint vectorCount,
    const jint dimension)
{
    try
    {
        if (vectorCount <= 0)
        {
            throw std::invalid_argument(
                "vectorCount must be positive."
            );
        }

        if (dimension <= 0)
        {
            throw std::invalid_argument(
                "dimension must be positive."
            );
        }

        const std::size_t expectedElements =
            checkedMultiply(
                static_cast<std::size_t>(
                    vectorCount
                ),
                static_cast<std::size_t>(
                    dimension
                ),
                "Database dimensions overflow size_t."
            );

        std::vector<float> database =
            copyFloatArray(
                environment,
                databaseArray,
                expectedElements,
                "database"
            );

        if (environment->ExceptionCheck())
        {
            return 0;
        }

        auto nativeIndex =
            std::make_shared<
                NativeIndex
            >(
                database,
                vectorCount,
                dimension
            );

        jlong handle =
            nextHandle.fetch_add(
                1,
                std::memory_order_relaxed
            );

        if (handle <= 0)
        {
            throw std::runtime_error(
                "Native handle space exhausted."
            );
        }

        {
            std::lock_guard<std::mutex>
                lock(
                    registryMutex
                );

            registry.emplace(
                handle,
                std::move(
                    nativeIndex
                )
            );
        }

        return handle;
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );

        return 0;
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );

        return 0;
    }
    catch (...)
    {
        throwIllegalState(
            environment,
            "Unknown native Vortex CUDA failure while creating the index."
        );

        return 0;
    }
}

JNIEXPORT void JNICALL
Java_com_vortex_cuda_VortexNative_nativeDestroyIndex(
    JNIEnv* environment,
    jclass,
    const jlong handle)
{
    translateVoidExceptions(
        environment,
        [&]()
        {
            if (handle <= 0)
            {
                return;
            }

            std::shared_ptr<NativeIndex>
                removed;

            {
                std::lock_guard<std::mutex>
                    lock(
                        registryMutex
                    );

                const auto iterator =
                    registry.find(
                        handle
                    );

                if (iterator ==
                    registry.end())
                {
                    return;
                }

                removed =
                    std::move(
                        iterator->second
                    );

                registry.erase(
                    iterator
                );
            }

            // Destruction occurs outside the global registry lock.
            // If a search already owns a shared_ptr, the CUDA index
            // stays alive until that search finishes.
            removed.reset();
        }
    );
}

JNIEXPORT void JNICALL
Java_com_vortex_cuda_VortexNative_nativeSearch(
    JNIEnv* environment,
    jclass,
    const jlong handle,
    jfloatArray queryArray,
    const jint queryCount,
    const jint topK,
    const jint precision,
    jintArray outputIndices,
    jfloatArray outputDistances)
{
    translateVoidExceptions(
        environment,
        [&]()
        {
            const std::shared_ptr<NativeIndex>
                nativeIndex =
                    requireIndex(
                        handle
                    );

            if (queryCount <= 0 ||
                queryCount >
                    GpuVectorIndex::
                        MAX_BATCH_SIZE)
            {
                throw std::invalid_argument(
                    "queryCount must be between 1 and 32."
                );
            }

            if (topK <= 0 ||
                topK >
                    GpuVectorIndex::
                        MAX_TOP_K ||
                topK >
                    nativeIndex->index
                        .vectorCount())
            {
                throw std::invalid_argument(
                    "topK must be between 1 and min(vectorCount, 32)."
                );
            }

            const std::size_t queryElements =
                checkedMultiply(
                    static_cast<std::size_t>(
                        queryCount
                    ),
                    static_cast<std::size_t>(
                        nativeIndex->index
                            .dimension()
                    ),
                    "Query dimensions overflow size_t."
                );

            const std::size_t resultElements =
                checkedMultiply(
                    static_cast<std::size_t>(
                        queryCount
                    ),
                    static_cast<std::size_t>(
                        topK
                    ),
                    "Result dimensions overflow size_t."
                );

            const std::vector<float> queries =
                copyFloatArray(
                    environment,
                    queryArray,
                    queryElements,
                    "queries"
                );

            if (environment->ExceptionCheck())
            {
                return;
            }

            validateOutputArray(
                environment,
                outputIndices,
                resultElements,
                "outputIndices"
            );

            validateOutputArray(
                environment,
                outputDistances,
                resultElements,
                "outputDistances"
            );

            GpuBatchTopKResult result;

            {
                std::lock_guard<std::mutex>
                    searchLock(
                        nativeIndex->
                            searchMutex
                    );

                result =
                    executeSearch(
                        *nativeIndex,
                        queries,
                        queryCount,
                        topK,
                        precision
                    );
            }

            if (result.neighbors.size() !=
                resultElements)
            {
                throw std::runtime_error(
                    "Native search returned an unexpected result size."
                );
            }

            std::vector<jint> indices(
                resultElements
            );

            std::vector<jfloat> distances(
                resultElements
            );

            for (std::size_t index = 0;
                 index <
                     resultElements;
                 ++index)
            {
                indices[index] =
                    static_cast<jint>(
                        result.neighbors[
                            index
                        ].index
                    );

                distances[index] =
                    static_cast<jfloat>(
                        result.neighbors[
                            index
                        ].distance
                    );
            }

            if (!indices.empty())
            {
                environment->SetIntArrayRegion(
                    outputIndices,
                    0,
                    static_cast<jsize>(
                        indices.size()
                    ),
                    indices.data()
                );

                if (environment->ExceptionCheck())
                {
                    return;
                }

                environment->SetFloatArrayRegion(
                    outputDistances,
                    0,
                    static_cast<jsize>(
                        distances.size()
                    ),
                    distances.data()
                );
            }
        }
    );
}

JNIEXPORT jlong JNICALL
Java_com_vortex_cuda_VortexNative_nativeFp32IndexBytes(
    JNIEnv* environment,
    jclass,
    const jlong handle)
{
    try
    {
        const auto nativeIndex =
            requireIndex(
                handle
            );

        return
            static_cast<jlong>(
                nativeIndex->index
                    .indexBytes()
            );
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );

        return 0;
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );

        return 0;
    }
}

JNIEXPORT jlong JNICALL
Java_com_vortex_cuda_VortexNative_nativeFp16IndexBytes(
    JNIEnv* environment,
    jclass,
    const jlong handle)
{
    try
    {
        const auto nativeIndex =
            requireIndex(
                handle
            );

        return
            static_cast<jlong>(
                nativeIndex->index
                    .fp16IndexBytes()
            );
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );

        return 0;
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );

        return 0;
    }
}

JNIEXPORT jdouble JNICALL
Java_com_vortex_cuda_VortexNative_nativeFp32UploadTimeMs(
    JNIEnv* environment,
    jclass,
    const jlong handle)
{
    try
    {
        const auto nativeIndex =
            requireIndex(
                handle
            );

        return
            static_cast<jdouble>(
                nativeIndex->index
                    .indexUploadTimeMs()
            );
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );

        return 0.0;
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );

        return 0.0;
    }
}

JNIEXPORT jdouble JNICALL
Java_com_vortex_cuda_VortexNative_nativeFp16BuildTimeMs(
    JNIEnv* environment,
    jclass,
    const jlong handle)
{
    try
    {
        const auto nativeIndex =
            requireIndex(
                handle
            );

        return
            static_cast<jdouble>(
                nativeIndex->index
                    .fp16IndexBuildTimeMs()
            );
    }
    catch (
        const std::invalid_argument& exception
    )
    {
        throwIllegalArgument(
            environment,
            exception.what()
        );

        return 0.0;
    }
    catch (
        const std::exception& exception
    )
    {
        throwIllegalState(
            environment,
            exception.what()
        );

        return 0.0;
    }
}


JNIEXPORT jstring JNICALL
Java_com_vortex_cuda_VortexNative_nativeDeviceName(
    JNIEnv* environment,
    jclass)
{
    try
    {
        int device = 0;
        cudaDeviceProp properties{};

        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(
            cudaGetDeviceProperties(
                &properties,
                device
            )
        );

        return environment->NewStringUTF(
            properties.name
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );

        return nullptr;
    }
}

JNIEXPORT jint JNICALL
Java_com_vortex_cuda_VortexNative_nativeComputeCapabilityMajor(
    JNIEnv* environment,
    jclass)
{
    try
    {
        int device = 0;
        cudaDeviceProp properties{};

        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(
            cudaGetDeviceProperties(
                &properties,
                device
            )
        );

        return static_cast<jint>(
            properties.major
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );
        return 0;
    }
}

JNIEXPORT jint JNICALL
Java_com_vortex_cuda_VortexNative_nativeComputeCapabilityMinor(
    JNIEnv* environment,
    jclass)
{
    try
    {
        int device = 0;
        cudaDeviceProp properties{};

        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(
            cudaGetDeviceProperties(
                &properties,
                device
            )
        );

        return static_cast<jint>(
            properties.minor
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );
        return 0;
    }
}

JNIEXPORT jlong JNICALL
Java_com_vortex_cuda_VortexNative_nativeCudaFreeMemoryBytes(
    JNIEnv* environment,
    jclass)
{
    try
    {
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;

        CUDA_CHECK(
            cudaMemGetInfo(
                &freeBytes,
                &totalBytes
            )
        );

        return static_cast<jlong>(
            freeBytes
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );
        return 0;
    }
}

JNIEXPORT jlong JNICALL
Java_com_vortex_cuda_VortexNative_nativeCudaTotalMemoryBytes(
    JNIEnv* environment,
    jclass)
{
    try
    {
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;

        CUDA_CHECK(
            cudaMemGetInfo(
                &freeBytes,
                &totalBytes
            )
        );

        return static_cast<jlong>(
            totalBytes
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );
        return 0;
    }
}

JNIEXPORT jint JNICALL
Java_com_vortex_cuda_VortexNative_nativeCudaRuntimeVersion(
    JNIEnv* environment,
    jclass)
{
    try
    {
        int version = 0;
        CUDA_CHECK(
            cudaRuntimeGetVersion(
                &version
            )
        );

        return static_cast<jint>(
            version
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );
        return 0;
    }
}

JNIEXPORT jint JNICALL
Java_com_vortex_cuda_VortexNative_nativeCudaDriverVersion(
    JNIEnv* environment,
    jclass)
{
    try
    {
        int version = 0;
        CUDA_CHECK(
            cudaDriverGetVersion(
                &version
            )
        );

        return static_cast<jint>(
            version
        );
    }
    catch (const std::exception& exception)
    {
        throwIllegalState(
            environment,
            exception.what()
        );
        return 0;
    }
}

} // extern "C"
