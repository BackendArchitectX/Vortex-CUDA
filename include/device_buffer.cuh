#pragma once

#include "cuda_check.cuh"

#include <cstddef>
#include <utility>

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(const std::size_t count)
        : count_(count)
    {
        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                count_ * sizeof(T)
            )
        );
    }

    ~DeviceBuffer()
    {
        if (data_ != nullptr)
        {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(
        const DeviceBuffer&
    ) = delete;

    DeviceBuffer(
        DeviceBuffer&& other
    ) noexcept
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

    DeviceBuffer& operator=(
        DeviceBuffer&& other
    ) noexcept
    {
        if (this != &other)
        {
            if (data_ != nullptr)
            {
                cudaFree(data_);
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