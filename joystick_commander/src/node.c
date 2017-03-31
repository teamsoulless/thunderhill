/**
 * @file node.c
 * @brief Node Layer Source.
 *
 * Joystick device: Logitech Gamepad F310
 * Mode switch (on back of controller): set to mode X
 * Front mode button: set to off (LED is off)
 * Brake controls: left trigger
 * Throttle controls: right trigger
 * Steering controls: right stick
 * Left turn signal: left trigger button
 * Right turn signal: right trigger button
 * Shift to drive: 'A' button
 * Shift to park: 'Y' button
 * Shift to neutral: 'X' button
 * Shift to reverse: 'B' button
 * Enable controls: 'start' button
 * Disable controls: 'back' button
 *
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>

#include "polysync_core.h"
#include "polysync_sdf.h"
#include "polysync_node.h"
#include "polysync_message.h"
#include "polysync_node_template.h"

#include "joystick.h"
#include "messages.h"
#include "commander.h"




// *****************************************************
// static global types/macros
// *****************************************************

/**
 * @brief node update and publish interval. [microseconds]
 *
 * Defines the update and publish rate of the node.
 *
 * 50,000 us == 50 ms == 20 Hertz
 *
 */
#define NODE_UPDATE_INTERVAL (50000)


/**
 * @brief Node sleep interval. [microseconds]
 *
 * Specifies the amount of time to sleep for during each wait/sleep cycle.
 *
 * This prevents our node from overloading the host.
 *
 */
#define NODE_SLEEP_TICK_INTERVAL (1000)



// *****************************************************
// static global data
// *****************************************************

/**
 * @brief Warning string.
 *
 */
static const char WARNING_STRING[] =
"\nWARNING: example is built for "
"the Joystick device: Logitech Gamepad F310\n"
"Back mode switch: 'X' setting\n"
"Front mode button: off (LED is off)"
"Brake controls: left trigger\n"
"Throttle controls: right trigger\n"
"Steering controls: right stick\n"
"Left turn signal: left trigger button\n"
"Right turn signal: right trigger button\n"
"Shift to drive: 'A' button\n"
"Shift to park: 'Y' button\n"
"Shift to neutral: 'X' button\n"
"Shift to reverse: 'B' button\n"
"Enable controls: 'start' button\n"
"Disable controls: 'back' button\n\n";




// *****************************************************
// static declarations
// *****************************************************


/**
 * @brief Node template set configuration callback function.
 *
 * If the host provides command line arguments they will be set, and available
 * for parsing (ie getopts).
 *
 * \li "on_init" - Called once after node transitions into the INIT state.
 * \li "on_release" - Called once on node exit.
 * \li "on_warn" - Called continously while in WARN state.
 * \li "on_error" - Called continously while in ERROR state.
 * \li "on_fatal" - Called once after node transitions into the FATAL state before terminating.
 * \li "on_ok" - Called continously while in OK state.
 *
 * @note Returning a DTC other than DTC_NONE will cause the node to transition
 * into the fatal state and terminate.
 *
 * @param [in] node_config A pointer to \ref ps_node_configuration_data which specifies the configuration.
 *
 * @return DTC code:
 * \li \ref DTC_NONE (zero) if success.
 *
 */
static int set_configuration(
        ps_node_configuration_data * const node_config );


/**
 * @brief Node template on_init callback function.
 *
 * Called once after node transitions into the INIT state.
 *
 * @param [in] node_ref Node reference, provided by node template API.
 * @param [in] state A pointer to \ref ps_diagnostic_state which stores the current state of the node.
 * @param [in] user_data A pointer to user data, provided by user during configuration.
 *
 */
static void on_init(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data );


/**
 * @brief Node template on_release callback function.
 *
 * Called once on node exit.
 *
 * @param [in] node_ref Node reference, provided by node template API.
 * @param [in] state A pointer to \ref ps_diagnostic_state which stores the current state of the node.
 * @param [in] user_data A pointer to user data, provided by user during configuration.
 *
 */
static void on_release(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data );


/**
 * @brief Node template on_error callback function.
 *
 * Called continously while in ERROR state.
 *
 * @param [in] node_ref Node reference, provided by node template API.
 * @param [in] state A pointer to \ref ps_diagnostic_state which stores the current state of the node.
 * @param [in] user_data A pointer to user data, provided by user during configuration.
 *
 */
static void on_error(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data );


/**
 * @brief Node template on_fatal callback function.
 *
 * Called once after node transitions into the FATAL state before terminating.
 *
 * @param [in] node_ref Node reference, provided by node template API.
 * @param [in] state A pointer to \ref ps_diagnostic_state which stores the current state of the node.
 * @param [in] user_data A pointer to user data, provided by user during configuration.
 *
 */
static void on_fatal(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data );


/**
 * @brief Node template on_warn callback function.
 *
 * Called continously while in WARN state.
 *
 * @param [in] node_ref Node reference, provided by node template API.
 * @param [in] state A pointer to \ref ps_diagnostic_state which stores the current state of the node.
 * @param [in] user_data A pointer to user data, provided by user during configuration.
 *
 */
static void on_warn(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data );


/**
 * @brief Node template on_ok callback function.
 *
 * Called continously while in OK state.
 *
 * @param [in] node_ref Node reference, provided by node template API.
 * @param [in] state A pointer to \ref ps_diagnostic_state which stores the current state of the node.
 * @param [in] user_data A pointer to user data, provided by user during configuration.
 *
 */
static void on_ok(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data );


//
static ps_timestamp get_time_since(
        const ps_timestamp past,
        ps_timestamp * const now );


/**
 * @brief Commander update loop.
 *
 * Called by on_warn and on_ok.
 *
 * @param [in] node_ref Node reference.
 * @param [in] commander A pointer to \ref commander_s which specifies the configuration.
 *
 */
static void update_loop(
        ps_node_ref const node_ref,
        commander_s * const commander );




// *****************************************************
// static definitions
// *****************************************************

//
static int set_configuration(
        ps_node_configuration_data * const node_config )
{
    const char default_node_name[] = "polysync-joystick-commander";


    // show warning string
    printf( WARNING_STRING );

    // set defaults
    // node type
    node_config->node_type = PSYNC_NODE_TYPE_API_USER;

    // set node domain
    node_config->domain_id = PSYNC_DEFAULT_DOMAIN;

    // set node SDF key
    node_config->sdf_key = PSYNC_SDF_ID_INVALID;

    // set node flags
    node_config->flags = 0;

    // set user data
    node_config->user_data = NULL;

    // set node name
    memset( node_config->node_name, 0, sizeof(node_config->node_name) );
    strncpy( node_config->node_name, default_node_name, sizeof(node_config->node_name) );

    // create node data
    commander_s * const commander = calloc( 1, sizeof(*commander) );
    if( commander == NULL )
    {
        psync_log_error( "failed to create node data" );
        return DTC_MEMERR;
    }

    // set user data pointer to our top-level node data
    // this will get passed around to the various interface routines
    node_config->user_data = (void*) commander;


    return DTC_NONE;
}


//
static void on_init(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data )
{
    int ret = DTC_NONE;
    commander_s * const commander = (commander_s*) user_data;

    // check reference since other routines don't
    if( commander == NULL )
    {
        psync_log_error( "invalid node data" );
        psync_node_activate_fault( node_ref, DTC_USAGE, NODE_STATE_FATAL );
        return;
    }

    // zero
    commander->dest_control_node_guid = PSYNC_GUID_INVALID;

    // reset discovered GUID
    ret = messages_reset_discovered_guid();
    if( ret != DTC_NONE )
    {
        psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        return;
    }

    // allocate messages
    ret = messages_alloc( node_ref, &commander->messages );
    if( ret != DTC_NONE )
    {
        psync_log_error( "failed to allocate PolySync messages - check data model and installation files" );
        psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        return;
    }

    // set safe state
    ret = commander_set_safe( commander );
    if( ret != DTC_NONE )
    {
        psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        return;
    }

    // register a listener to PolySync response messages
    ret = messages_register_response_subscriber(
            node_ref,
            &commander->messages );
    if( ret != DTC_NONE )
    {
        psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        return;
    }

    // send command to enumerate all supported dynamic driver nodes
    ret = commander_enumerate_control_nodes(
            node_ref,
            commander );
    if( ret != DTC_NONE )
    {
        psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        return;
    }

    // init joystick subsystem
    ret = jstick_init_subsystem();
    if( ret != DTC_NONE )
    {
        psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        return;
    }

    // get number of joysticks visible
    const int num_joysticks = jstick_get_num_devices();

    // if any are available
    if( num_joysticks > 0 )
    {
        // device GUID
        joystick_guid_s js_guid;

        // default device
        const unsigned long default_device_index = 0;

        // get GUID of device at index
        ret = jstick_get_guid_at_index( 0, &js_guid );
        if( ret != DTC_NONE )
        {
            psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
            return;
        }

        psync_log_warn(
                "found %d devices -- connecting to device at system index %lu - GUID: %s",
                num_joysticks,
                default_device_index,
                js_guid.ascii_string );

        // connect to first device
        ret = jstick_open(
                default_device_index,
                &commander->joystick );
        if( ret != DTC_NONE )
        {
            psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
            return;
        }

        // wait for safe state
        psync_log_info( "waiting for joystick controls to zero" );
        do
        {
            ret = commander_check_for_safe_joystick( commander );

            if( ret == DTC_UNAVAILABLE )
            {
                // wait a little for the next try
                (void) psync_sleep_micro( NODE_UPDATE_INTERVAL );
            }
            else if( ret != DTC_NONE )
            {
                psync_log_error( "failed to wait for joystick to zero the control values" );
                psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );

                // return now
                break;
            }
        }
        while( ret != DTC_NONE );
    }
    else
    {
        psync_log_error( "no joystick/devices available on the host" );
        psync_node_activate_fault( node_ref, DTC_USAGE, NODE_STATE_FATAL );

        // return now
        return;
    }

    // wait for a valid node to respond in the normal on-ok logic
    psync_log_info( "waiting for a PolySync control node to respond" );
}


//
static void on_release(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data )
{
    commander_s *commander = (commander_s*) user_data;


    if( commander != NULL )
    {
        // send command to disable controls
        (void) commander_disable_controls( node_ref, commander );

        // set commander safe state
        (void) commander_set_safe( commander );

        // close device if needed
        jstick_close( &commander->joystick );

        // free messages
        (void) messages_free(
                node_ref,
                &commander->messages );

        free( commander );
        commander = NULL;
    }

    // release joystick subsystem
    jstick_release_subsystem();
}


//
static void on_error(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data )
{
    commander_s * const commander = (commander_s*) user_data;


    if( commander != NULL )
    {
        ps_timestamp now = 0;

        // get the amount of time since our last update/publish
        const ps_timestamp time_since_last_publish =
                get_time_since( commander->last_commander_update, &now );

        // only update/publish at our defined interval
        if( time_since_last_publish >= NODE_UPDATE_INTERVAL )
        {
            // send disable controls command
            (void) commander_disable_controls(
                    node_ref,
                    commander );

            // send e-stop commands
            (void) commander_estop(
                    node_ref,
                    commander );

            commander->last_commander_update = now;
        }
    }
}


//
static void on_fatal(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data )
{
    // cast node data
    commander_s * const commander = (commander_s*) user_data;


    // if node data valid
    if( commander != NULL )
    {
        // send disable controls command
        (void) commander_disable_controls(
                node_ref,
                commander );

        // send e-stop commands
        (void) commander_estop(
                node_ref,
                commander );
    }
}


//
static void on_warn(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data )
{
    // cast node data
    commander_s * const commander = (commander_s*) user_data;


    // check if command data is valid
    if( commander_is_valid( commander ) != DTC_NONE )
    {
        // activate DTC
        psync_node_activate_fault( node_ref, DTC_DATAERR, NODE_STATE_FATAL );

        // return now
        return;
    }

    // do update loop
    update_loop( node_ref, commander );
}


//
static void on_ok(
        ps_node_ref const node_ref,
        const ps_diagnostic_state * const state,
        void * const user_data )
{
    // cast node data
    commander_s * const commander = (commander_s*) user_data;


    // check if command data is valid
    if( commander_is_valid( commander ) != DTC_NONE )
    {
        // activate DTC
        psync_node_activate_fault( node_ref, DTC_DATAERR, NODE_STATE_FATAL );

        // return now
        return;
    }

    // do update loop
    update_loop( node_ref, commander );
}


//
static ps_timestamp get_time_since(
        const ps_timestamp past,
        ps_timestamp * const now )
{
    ps_timestamp delta = 0;
    ps_timestamp m_now = 0;


    const int ret = psync_get_timestamp( &m_now );

    if( ret == DTC_NONE )
    {
        if( m_now >= past )
        {
            delta = (m_now - past);
        }
    }

    // update provided argument if valid
    if( now != NULL )
    {
        (*now) = m_now;
    }


    return delta;
}


//
static void update_loop(
        ps_node_ref const node_ref,
        commander_s * const commander )
{
    int ret = DTC_NONE;
    ps_timestamp now = 0;


    // get the amount of time since our last update/publish
    const ps_timestamp time_since_last_publish =
            get_time_since( commander->last_commander_update, &now );

    // if we have a valid destination node GUID
    if( commander->dest_control_node_guid != PSYNC_GUID_INVALID )
    {
        // only update/publish at our defined interval
        if( time_since_last_publish >= NODE_UPDATE_INTERVAL )
        {
            // update commander, send command messages
            ret = commander_update(
                    node_ref,
                    commander );
            if( ret != DTC_NONE )
            {
                psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
            }

            commander->last_commander_update = now;
        }
    }
    else
    {
        ret = messages_get_discovered_guid(
                &commander->dest_control_node_guid );
        if( ret != DTC_NONE )
        {
            psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
        }

        if( commander->dest_control_node_guid != PSYNC_GUID_INVALID )
        {
            psync_log_info( "got valid response from control node GUID 0x%016llX (%llu)",
                    (unsigned long long) commander->dest_control_node_guid,
                    (unsigned long long) commander->dest_control_node_guid );

            ret = messages_unregister_response_subscriber(
                    node_ref,
                    &commander->messages );
            if( ret != DTC_NONE )
            {
                psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
            }
        }
        else
        {
            // check for re-enumerate button action
            ret = commander_check_reenumeration(
                    node_ref,
                    commander );
            if( ret != DTC_NONE )
            {
                psync_node_activate_fault( node_ref, ret, NODE_STATE_FATAL );
            }
        }
    }

    // sleep for 1 ms to avoid loading the CPU
    (void) psync_sleep_micro( NODE_SLEEP_TICK_INTERVAL );
}




// *****************************************************
// public definitions
// *****************************************************

//
int main( int argc, char **argv )
{
    // callback data
    ps_node_callbacks callbacks;


    // zero
    memset( &callbacks, 0, sizeof(callbacks) );

    // set callbacks
    callbacks.set_config = &set_configuration;
    callbacks.on_init = &on_init;
    callbacks.on_release = &on_release;
    callbacks.on_warn = &on_warn;
    callbacks.on_error = &on_error;
    callbacks.on_fatal = &on_fatal;
    callbacks.on_ok = &on_ok;


    // use PolySync main entry, this will give execution context to node template machine
    return( psync_node_main_entry( &callbacks, argc, argv ) );
}
