cmake_minimum_required( VERSION 2.8 )

project( recorder-node )

if( NOT PSYNC_HOME )
    if( $ENV{PSYNC_HOME} )
        set( PSYNC_HOME $ENV{PSYNC_HOME} )
    else()
        set( PSYNC_HOME /usr/local/polysync )
    endif()
endif()

include( ${PSYNC_HOME}/BuildResources.cmake )

include_directories(
    ${PSYNC_INCLUDE_DIRS}
)

add_executable( ${PROJECT_NAME}
    RecorderNode.cpp
)

target_link_libraries( ${PROJECT_NAME}
    ${PSYNC_LIBS}
)
