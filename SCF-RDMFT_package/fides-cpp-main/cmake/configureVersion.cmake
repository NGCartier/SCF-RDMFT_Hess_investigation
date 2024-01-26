include(${CMAKE_CURRENT_LIST_DIR}/version.cmake)
set(AMICI_VERSION ${PROJECT_VERSION})
get_filename_component(directory ${DST} DIRECTORY)
file(MAKE_DIRECTORY ${directory})
configure_file(${SRC} ${DST} @ONLY)