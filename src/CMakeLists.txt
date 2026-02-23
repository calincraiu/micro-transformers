# core
set(CORE_HEADERS 
    "${CMAKE_SOURCE_DIR}/include/core/Value.h"
    "${CMAKE_SOURCE_DIR}/include/core/Node.h"
    "${CMAKE_SOURCE_DIR}/include/core/Tape.h"
)
set(CORE_SOURCES
    core/Value.cpp
    core/Tape.cpp
)
add_library(core
    ${CORE_SOURCES} 
    ${CORE_HEADERS}
)
target_include_directories(core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

# nn
set(NN_HEADERS 
    "${CMAKE_SOURCE_DIR}/include/nn/Linear.h"
    "${CMAKE_SOURCE_DIR}/include/nn/MLP.h"
)
set(NN_SOURCES
    nn/Linear.cpp
    nn/MLP.cpp
)
add_library(nn
    ${NN_SOURCES} 
    ${NN_HEADERS}
)
target_include_directories(nn PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)