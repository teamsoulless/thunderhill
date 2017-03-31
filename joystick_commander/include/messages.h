/**
 * @file messages.h
 * @brief Message Utilities Interface.
 *
 */




#ifndef MESSAGES_H
#define MESSAGES_H




#include "polysync_core.h"




/**
 * @brief Disable controls command identifier.
 *
 * Sensor specific command handled by the DataSpeed MKZ interface.
 *
 */
#define DISABLE_CONTROLS_COMMAND_ID (5000)


/**
 * @brief Enable controls command identifier.
 *
 * Sensor specific command handled by the DataSpeed MKZ interface.
 *
 */
#define ENABLE_CONTROLS_COMMAND_ID (5001)


/**
 * @brief PolySync sensor type value for the DataSpeed MKZ.
 *
 */
#define SENSOR_TYPE_DATASPEED_MKZ (300)




/**
 * @brief Message set.
 *
 * Container for PolySync messages.
 *
 */
typedef struct
{
    //
    //
    ps_platform_brake_command_msg *brake_cmd; /*!< Platform brake command message. */
    //
    //
    ps_platform_throttle_command_msg *throttle_cmd; /*!< Platform throttle command message. */
    //
    //
    ps_platform_steering_command_msg *steering_cmd; /*!< Platform steering wheel command message. */
    //
    //
    ps_platform_gear_command_msg *gear_cmd; /*!< Platform gear command message. */
    //
    //
    ps_platform_turn_signal_command_msg *turn_signal_cmd; /*!< Platform turn signal command message. */
    //
    //
    ps_command_msg *command_msg; /*!< Command message. */
    //
    //
    ps_msg_type response_msg_type; /*!< PolySync response message type identifer.
                                    * Used to register a subscriber to response messages. */
} messages_s;



//
int messages_alloc(
        ps_node_ref node_ref,
        messages_s * const messages );


//
int messages_register_response_subscriber(
        ps_node_ref node_ref,
        messages_s * const messages );


//
int messages_unregister_response_subscriber(
        ps_node_ref node_ref,
        messages_s * const messages );


//
int messages_free(
        ps_node_ref node_ref,
        messages_s * const messages );


//
int messages_is_valid(
        const messages_s * const const messages );


//
int messages_set_default_values(
        const ps_guid dest_guid,
        messages_s * const messages );


//
int messages_get_discovered_guid(
        ps_guid * const guid );


//
int messages_reset_discovered_guid( void );



#endif	/* MESSAGES_H */
