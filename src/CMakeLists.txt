# Define the headers explicitly so IDEs like Visual Studio show them
set(CORE_HEADERS 
    "${CMAKE_SOURCE_DIR}/include/core/Value.h"
)

set(CORE_SOURCES
    core/Value.cpp
)

# Create the library
add_library(core
    ${CORE_SOURCES} 
    ${CORE_HEADERS}
)

# Tell CMake where the headers for this library are.
target_include_directories(core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)