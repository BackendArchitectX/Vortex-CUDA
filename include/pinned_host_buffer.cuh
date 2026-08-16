#pragma once

#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

template <typename T>
class PinnedHostBuffer
{
public:
    explicit PinnedHostBuffer(
        const std::size_t count)
        : count_(count)
    {
        if (count_ == 0)
        {
            return;
        }

        CUDA_CHECK(
            cudaHostAlloc(
                reinterpret_cast<void**>(
                    &data_
                ),
                count_ *
                    sizeof(T),
                cudaHostAllocDefault
            )
        );
    }

    ~PinnedHostBuffer()
    {
        if (data_ != nullptr)
        {
            cudaFreeHost(
                data_
            );
        }
    }

    PinnedHostBuffer(
        const PinnedHostBuffer&
    ) = delete;

    PinnedHostBuffer& operator=(
        const PinnedHostBuffer&
    ) = delete;

    PinnedHostBuffer(
        PinnedHostBuffer&& other) noexcept
        : data_(
            std::exchange(
                other.data_,
                nullptr
            )
        ),
          count_(
            std::exchange(
                other.count_,
                0
            )
        )
    {
    }

    PinnedHostBuffer& operator=(
        PinnedHostBuffer&& other) noexcept
    {
        if (this !=
            &other)
        {
            if (data_ != nullptr)
            {
                cudaFreeHost(
                    data_
                );
            }

            data_ =
                std::exchange(
                    other.data_,
                    nullptr
                );

            count_ =
                std::exchange(
                    other.count_,
                    0
                );
        }

        return *this;
    }

    [[nodiscard]]
    T* data() noexcept
    {
        return data_;
    }

    [[nodiscard]]
    const T* data() const noexcept
    {
        return data_;
    }

    [[nodiscard]]
    std::size_t size() const noexcept
    {
        return count_;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};
