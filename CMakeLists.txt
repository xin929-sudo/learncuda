cmake_minimum_required(VERSION 3.15)

# 项目名称
project(CudaCourse LANGUAGES CXX CUDA)

# =======================================================================
# 🚀 修改点 1：只设置宿主 C++ 编译器的标准为 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 🚀 修改点 2：删掉 set(CMAKE_CUDA_STANDARD 17)，改为强行给 nvcc 传参！
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# 🚀 修改点 3：加入 NVIDIA 官方提示的宏，双重屏蔽报错
add_compile_definitions(CCCL_IGNORE_DEPRECATED_CPP_DIALECT)
# =======================================================================

# ================= 配置区域 =================

# 1. 设置显卡架构 (RTX 2060 = sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# 2. 设置输出目录 (把 .exe 都生成到 bin 文件夹，不乱放)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 3. 解决中文乱码问题 (强制 UTF-8)
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

# ================= 自动编译逻辑 =================

# 定义一个函数：传入文件夹名，自动生成可执行文件
function(add_day_project DIR_NAME)
    # 查找该文件夹下所有的 .cu 和 .cpp 文件
    file(GLOB SOURCES "${DIR_NAME}/*.cu" "${DIR_NAME}/*.cpp")
    
    # 如果找到了源文件
    if(SOURCES)
        # 创建可执行文件 (名字就是文件夹名，例如 day1.exe)
        add_executable(${DIR_NAME} ${SOURCES})
        
        # 包含该文件夹作为头文件路径
        target_include_directories(${DIR_NAME} PRIVATE ${DIR_NAME})
        
        message(STATUS "✅ 已添加构建目标: ${DIR_NAME}")
    else()
        message(WARNING "⚠️  文件夹 ${DIR_NAME} 中没有找到源码！")
    endif()
endfunction()

# ================= 在这里注册你的文件夹 =================

# add_day_project(day1)
# add_day_project(day2)
add_day_project(day8)