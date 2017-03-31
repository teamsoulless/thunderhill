/**
 * @file joystick.c
 * @brief Joystick Interface Source.
 *
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_joystick.h>

#include "polysync_core.h"
#include "joystick.h"




// *****************************************************
// static global types/macros
// *****************************************************

/**
 * @brief Button press debounce delay. [microseconds]
 *
 */
#define BUTTON_PRESSED_DELAY (5000)



// *****************************************************
// static global data
// *****************************************************




// *****************************************************
// static declarations
// *****************************************************




// *****************************************************
// static definitions
// *****************************************************




// *****************************************************
// public definitions
// *****************************************************

//
int jstick_init_subsystem( void )
{
    // local vars
    int ret = 0;


    // init joystick subsystem
    SDL_Init( SDL_INIT_JOYSTICK );

    // error check
    if( ret < 0 )
    {
        psync_log_error(
                "SDL_Init - %s",
                SDL_GetError() );
        return DTC_OSERR;
    }


    return DTC_NONE;
}


//
void jstick_release_subsystem( void )
{
    // release
    SDL_Quit();
}


//
int jstick_get_num_devices( void )
{
    // local vars
    int ret = 0;


    // get number of joysticks on the system
    ret = SDL_NumJoysticks();

    // error check
    if( ret < 0 )
    {
        psync_log_error(
                "SDL_NumJoysticks - %s",
                SDL_GetError() );
        return -1;
    }


    // return count
    return ret;
}


//
int jstick_get_guid_at_index(
        const unsigned long device_index,
        joystick_guid_s * const guid )
{
    if( guid == NULL )
    {
        return DTC_USAGE;
    }


    // get GUID
    const SDL_JoystickGUID m_guid =
            SDL_JoystickGetDeviceGUID( (int) device_index );

    // copy
    memcpy( guid->data, m_guid.data, sizeof(m_guid.data) );

    // get string representation
    memset( guid->ascii_string, 0, sizeof(guid->ascii_string) );
    SDL_JoystickGetGUIDString(
            m_guid,
            guid->ascii_string,
            sizeof(guid->ascii_string) );


    return DTC_NONE;
}


//
int jstick_open(
        const unsigned long device_index,
        joystick_device_s * const jstick )
{
    if( jstick == NULL )
    {
        return DTC_USAGE;
    }


    // open joystick at index
    jstick->handle = (void*) SDL_JoystickOpen( (int) device_index );

    // error check
    if( jstick->handle == JOYSTICK_DEVICE_HANDLE_INVALID )
    {
        psync_log_error(
                "SDL_JoystickOpen - %s",
                SDL_GetError() );
        return DTC_IOERR;
    }

    // get GUID
    const SDL_JoystickGUID m_guid =
            SDL_JoystickGetGUID( jstick->handle );

    // copy
    memcpy( jstick->guid.data, m_guid.data, sizeof(m_guid.data) );

    // get string representation
    memset( jstick->guid.ascii_string, 0, sizeof(jstick->guid.ascii_string) );
    SDL_JoystickGetGUIDString(
            m_guid,
            jstick->guid.ascii_string,
            sizeof(jstick->guid.ascii_string) );


    return DTC_NONE;
}


//
void jstick_close(
    joystick_device_s * const jstick )
{
    if( jstick == NULL )
    {
        return;
    }


    // if handle valid
    if( jstick->handle != JOYSTICK_DEVICE_HANDLE_INVALID )
    {
        // if attached
        if( SDL_JoystickGetAttached( jstick->handle ) == SDL_TRUE )
        {
            // close
            SDL_JoystickClose( jstick->handle );
        }

        // invalidate
        jstick->handle = JOYSTICK_DEVICE_HANDLE_INVALID;
    }
}


//
int jstick_update(
    joystick_device_s * const jstick )
{
    if( jstick == NULL )
    {
        return DTC_USAGE;
    }

    // local vars
    int ret = DTC_NONE;


    // if handle valid
    if( jstick->handle != JOYSTICK_DEVICE_HANDLE_INVALID )
    {
        // update
        SDL_JoystickUpdate();

        // check if attached
        if( SDL_JoystickGetAttached(jstick->handle) == SDL_FALSE )
        {
            psync_log_error( "SDL_JoystickGetAttached - device not attached" );

            // invalid handle
            ret = DTC_UNAVAILABLE;
        }
    }
    else
    {
        // invalid handle
        ret = DTC_UNAVAILABLE;
    }


    return ret;
}


//
int jstick_get_axis(
        joystick_device_s * const jstick,
        const unsigned long axis_index,
        int * const position )
{
    if( (jstick == NULL) || (position == NULL) )
    {
        return DTC_USAGE;
    }


    // zero
    (*position) = 0;

    // get axis value
    const Sint16 pos = SDL_JoystickGetAxis(
            jstick->handle,
            (int) axis_index );

    // convert
    (*position) = (int) pos;


    return DTC_NONE;
}


//
int jstick_get_button(
        joystick_device_s * const jstick,
        const unsigned long button_index,
        unsigned int * const state )
{
    if( (jstick == NULL) || (state == NULL) )
    {
        return DTC_USAGE;
    }


    // zero
    (*state) = JOYSTICK_BUTTON_STATE_NOT_PRESSED;

    // get button state
    const Uint8 m_state = SDL_JoystickGetButton(
            jstick->handle,
            (int) button_index );

    // convert
    if( m_state == 1 )
    {
        (*state) = JOYSTICK_BUTTON_STATE_PRESSED;
        (void) psync_sleep_micro( BUTTON_PRESSED_DELAY );
    }
    else
    {
        (*state) = JOYSTICK_BUTTON_STATE_NOT_PRESSED;
    }


    return DTC_NONE;
}


//
double jstick_normalize_axis_position(
        const int position,
        const double range_min,
        const double range_max )
{
    const double s = (double) position;
    const double a1 = (double) JOYSTICK_AXIS_POSITION_MIN;
    const double a2 = (double) JOYSTICK_AXIS_POSITION_MAX;
    const double b1 = range_min;
    const double b2 = range_max;


    // map value s in the range of a1 and a2, to t(return) in the range b1 and b2, linear
    return b1 + (s-a1) * (b2-b1) / (a2-a1);
}
