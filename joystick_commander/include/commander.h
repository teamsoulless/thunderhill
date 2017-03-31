/**
 * @file commander.h
 * @brief Commander Interface.
 *
 */




#ifndef COMMANDER_H
#define COMMANDER_H




#include "polysync_core.h"

#include "joystick.h"
#include "messages.h"




/**
 * @brief Maximum allowed throttle pedal position value. [normalized]
 *
 */
#define MAX_THROTTLE_PEDAL (0.3)


/**
 * @brief Maximum allowed brake pedal position value. [normalized]
 *
 */
#define MAX_BRAKE_PEDAL (0.8)


/**
 * @brief Minimum brake value to be considered enabled. [normalized]
 *
 * Throttle is disabled when brake value is greate than this value.
 *
 */
#define BRAKES_ENABLED_MIN (0.05)


/**
 * @brief Minimum allowed steering wheel angle value. [radians]
 *
 * Negative value means turning to the right.
 *
 */
#define MIN_STEERING_WHEEL_ANGLE (-M_PI * 2.0)


/**
 * @brief Maximum allowed steering wheel angle value. [radians]
 *
 * Positive value means turning to the left.
 *
 */
#define MAX_STEERING_WHEEL_ANGLE (M_PI * 2.0)


/**
 * @brief Maximum allowed absolute steering wheel angle rate value. [radians/second]
 *
 */
#define STEERING_WHEEL_ANGLE_RATE_LIMIT (M_PI_2)




/**
 * @brief Commander node data.
 *
 * Serves as a top-level container for the application's data structures.
 *
 */
typedef struct
{
    //
    //
    joystick_device_s joystick; /*!< Joystick handle. */
    //
    //
    messages_s messages; /*!< PolySync messages. */
    //
    //
    ps_timestamp last_commander_update; /*!< Last commander update timestamp. [microseconds] */
    //
    //
    ps_guid dest_control_node_guid; /*!< Node GUID that our control commmands are destined to.
                                     * Updated by the first response message received from a valid control node. */
} commander_s;




/**
 * @brief Wait for joystick throttle/brake values to be zero.
 *
 * @param [in] commander A pointer to \ref commander_s which specifies the joystick configuration.
 *
 * @return DTC code:
 * \li \ref DTC_NONE (zero) if joystick values safe.
 * \li \ref DTC_USAGE if arguments are invalid.
 * \li \ref DTC_CONFIG if configuration invalid.
 * \li \ref DTC_UNAVAILABLE if joystick values are not safe.
 *
 */
int commander_check_for_safe_joystick(
        commander_s * const commander );


//
int commander_is_valid(
        commander_s * const commander );


/**
 * @brief Set control command messages to their safe state.
 *
 * @param [in] commander A pointer to \ref commander_s which receives the safe state configuration.
 *
 * @return DTC code:
 * \li \ref DTC_NONE (zero) if success.
 *
 */
int commander_set_safe(
        commander_s * const commander );


//
int commander_enumerate_control_nodes(
        ps_node_ref node_ref,
        commander_s * const commander );


//
int commander_disable_controls(
        ps_node_ref node_ref,
        commander_s * const commander );


//
int commander_enable_controls(
        ps_node_ref node_ref,
        commander_s * const commander );


//
int commander_estop(
        ps_node_ref node_ref,
        commander_s * const commander );


//
int commander_check_reenumeration(
        ps_node_ref node_ref,
        commander_s * const commander );


//
int commander_update(
        ps_node_ref node_ref,
        commander_s * const commander );




#endif	/* COMMANDER_H */
