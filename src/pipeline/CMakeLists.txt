set(LUISA_RENDER_PIPELINE_SOURCES
        pipeline.cpp pipeline.h
        pipeline_data.cpp pipeline_data.h
        instance.h)

add_library(luisa-render-pipeline SHARED ${LUISA_RENDER_PIPELINE_SOURCES})
target_link_libraries(luisa-render-pipeline PUBLIC luisa-render-base)
set_target_properties(luisa-render-pipeline PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
