# Provides the imported target `wgpu_native`, the WebGPU implementation from
# https://github.com/gfx-rs/wgpu-native.
#
# A prebuilt release is downloaded unless WGPU_NATIVE_DIR points to an already
# extracted release (containing `include` and `lib`).

set(WGPU_NATIVE_VERSION "v29.0.1.1" CACHE STRING "wgpu-native release to download")
set(WGPU_NATIVE_DIR "" CACHE PATH "Path to an extracted wgpu-native release")

if(NOT WGPU_NATIVE_DIR)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(_wgpu_arch "aarch64")
    else()
        set(_wgpu_arch "x86_64")
    endif()

    if(WIN32)
        set(_wgpu_asset "wgpu-windows-${_wgpu_arch}-msvc-release")
    elseif(APPLE)
        set(_wgpu_asset "wgpu-macos-${_wgpu_arch}-release")
    else()
        set(_wgpu_asset "wgpu-linux-${_wgpu_arch}-release")
    endif()

    include(FetchContent)
    FetchContent_Declare(wgpu_native_release
        URL "https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_NATIVE_VERSION}/${_wgpu_asset}.zip"
    )
    FetchContent_MakeAvailable(wgpu_native_release)
    set(_wgpu_dir "${wgpu_native_release_SOURCE_DIR}")
else()
    set(_wgpu_dir "${WGPU_NATIVE_DIR}")
endif()

# Link statically, to not have to distribute the shared library
if(WIN32)
    set(_wgpu_lib "${_wgpu_dir}/lib/wgpu_native.lib")
else()
    set(_wgpu_lib "${_wgpu_dir}/lib/libwgpu_native.a")
endif()

if(NOT EXISTS "${_wgpu_lib}")
    message(FATAL_ERROR "Could not find wgpu-native library: ${_wgpu_lib}")
endif()

add_library(wgpu_native STATIC IMPORTED GLOBAL)
set_target_properties(wgpu_native PROPERTIES
    IMPORTED_LOCATION "${_wgpu_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${_wgpu_dir}/include/webgpu"
)

# System libraries required by the static library
if(WIN32)
    set_property(TARGET wgpu_native PROPERTY INTERFACE_LINK_LIBRARIES
        d3dcompiler ws2_32 userenv bcrypt ntdll opengl32 propsys runtimeobject
        oleaut32 ole32 dxgi d3d12 dxguid
    )
elseif(APPLE)
    set_property(TARGET wgpu_native PROPERTY INTERFACE_LINK_LIBRARIES
        "-framework Metal" "-framework QuartzCore" "-framework Foundation"
        "-framework CoreFoundation" "-framework IOKit" "-framework IOSurface"
    )
else()
    set_property(TARGET wgpu_native PROPERTY INTERFACE_LINK_LIBRARIES
        dl m pthread
    )
endif()
