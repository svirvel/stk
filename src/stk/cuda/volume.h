#pragma once

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/types.h>

namespace stk
{
    namespace cuda
    {
        template<typename T>
        struct VolumePtr
        {
            VolumePtr(const GpuVolume& vol) : 
                ptr((T*)vol.pitched_ptr().ptr),
                pitch(vol.pitched_ptr().pitch),
                ysize(vol.pitched_ptr().ysize),
                size(vol.size())
            {
                ASSERT(vol.voxel_type() == type_id<T>::id());
                ASSERT(vol.usage() == gpu::Usage_PitchedPointer);
            }

            __device__ T& operator()(int x, int y, int z)
            { 
                return ((T*)(((uint8_t*)ptr) + (y * pitch + z * pitch * ysize)))[x];
            }
            __device__ const T& operator()(int x, int y, int z) const
            {
                return ((T*)(((uint8_t*)ptr) + (y * pitch + z * pitch * ysize)))[x];
            }

            T* ptr;
            size_t pitch;
            size_t ysize;

            dim3 size;
        };

#ifdef __CUDACC__
        template<typename T>
        __device__ T linear_at_border(VolumePtr<T> vol, float x, float y, float z)
        {
            int x1 = int(floorf(x));
            int y1 = int(floorf(y));
            int z1 = int(floorf(z));

            int x2 = int(ceilf(x));
            int y2 = int(ceilf(y));
            int z2 = int(ceilf(z));

            if (x1 < 0 || x2 >= int(vol.size.x) ||
                y1 < 0 || y2 >= int(vol.size.y) ||
                z1 < 0 || z2 >= int(vol.size.z))
            {
                return 0.0f;
            }

            float xt = x - floorf(x);
            float yt = y - floorf(y);
            float zt = z - floorf(z);

            float s111 = vol(x1, y1, z1);
            float s211 = vol(x2, y1, z1);

            float s121 = vol(x1, y2, z1);
            float s221 = vol(x2, y2, z1);

            float s112 = vol(x1, y1, z2);
            float s212 = vol(x2, y1, z2);

            float s122 = vol(x1, y2, z2);
            float s222 = vol(x2, y2, z2);

            return T(
                (1 - zt) *
                (
                    (1 - yt) *
                    (
                        (1 - xt) * s111 +
                        (xt) * s211
                    ) +

                    (yt) *
                    (
                        (1 - xt) * s121 +
                        (xt) * s221
                    )
                ) +
            (zt) *
                (
                    (1 - yt)*
                    (
                        (1 - xt)*s112 +
                        (xt)*s212
                    ) +

                    (yt)*
                    (
                        (1 - xt)*s122 +
                        (xt)*s222
                    )
                )
            );
        }
#endif // __CUDACC__
    }
}