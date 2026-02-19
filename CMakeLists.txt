cmake_minimum_required(VERSION 3.15)

# 项目名称
project(CudaCourse LANGUAGES CXX CUDA)

# ================= 配置区域 =================

# 1. 设置显卡架构 (RTX 2060 = sm_75)
# 如果以后换 5090，改成 120 即可
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

# 以后写了 day2，把下面这行的注释(#)去掉就行了
# add_day_project(day2)
add_day_project(day5)