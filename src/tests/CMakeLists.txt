add_executable(test_distribution test_distribution.cpp)
target_link_libraries(test_distribution PRIVATE luisa::render)

add_executable(test_alias_method test_alias_method.cpp)
target_link_libraries(test_alias_method PRIVATE luisa::render)

add_executable(test_u64 test_u64.cpp)
target_link_libraries(test_u64 PRIVATE luisa::render)

# sky texture precomputation test
add_executable(test_sky test_sky.cpp)
target_link_libraries(test_sky PRIVATE luisa-render-texture-sky-precompute)
