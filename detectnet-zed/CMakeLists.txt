
file(GLOB detectnetZedSources *.cpp)
file(GLOB detectnetZedIncludes *.h )

cuda_add_executable(detectnet-zed ${detectnetZedSources})

target_link_libraries(detectnet-zed jetson-inference)

install(TARGETS detectnet-zed DESTINATION bin)
