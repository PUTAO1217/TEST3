import traci
import numpy as np

def get_reward():
    # intersection_queue = 0
    halt_n = traci.lane.getLastStepLength("-E1_0") + traci.lane.getLastStepLength(
        "-E1_1") + traci.lane.getLastStepLength("-E1_2") + traci.lane.getLastStepLength("-E1_3")
    halt_s = traci.lane.getLastStepLength("-E3_0") + traci.lane.getLastStepLength(
        "-E3_1") + traci.lane.getLastStepLength("-E3_2") + traci.lane.getLastStepLength("-E3_3")
    halt_e= traci.lane.getLastStepLength("-E2_0") + traci.lane.getLastStepLength(
        "-E2_1") + traci.lane.getLastStepLength("-E2_2") + traci.lane.getLastStepLength("-E2_3")
    halt_w= traci.lane.getLastStepLength("-E0_0") + traci.lane.getLastStepLength(
        "-E0_1") + traci.lane.getLastStepLength("-E0_2") + traci.lane.getLastStepLength("-E0_3")
    intersection_queue = halt_n + halt_s + halt_e + halt_w
    return intersection_queue

action_space = [0,1]


def get_custom_state():
    Position_Matrix = []
    Velocity_Matrix = []
    for i in range(16):
        Position_Matrix.append([])
        Velocity_Matrix.append([])
        for j in range(3):
            Position_Matrix[i].append(0)
            Velocity_Matrix[i].append(0)
    Position_Matrix = np.array(Position_Matrix)
    Velocity_Matrix = np.array(Velocity_Matrix)

    Loop1i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_1")
    Loop1i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_2")
    Loop1i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_3")
    Loop1i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_1")
    Loop1i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_2")
    Loop1i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_3")
    Loop1i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_1")
    Loop1i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_2")
    Loop1i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_3")
    Loop1i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_1")
    Loop1i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_2")
    Loop1i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_3")

    if len(Loop1i_0_1) != 0:
        Velocity_Matrix[0, 0] = traci.vehicle.getSpeed(Loop1i_0_1[0])
        Loop1i_0_1 = 1
    else:
        Loop1i_0_1 = 0

    if len(Loop1i_0_2) != 0:
        Velocity_Matrix[0, 1] = traci.vehicle.getSpeed(Loop1i_0_2[0])
        Loop1i_0_2 = 1
    else:
        Loop1i_0_2 = 0

    if len(Loop1i_0_3) != 0:
        Velocity_Matrix[0, 2] = traci.vehicle.getSpeed(Loop1i_0_3[0])
        Loop1i_0_3 = 1
    else:
        Loop1i_0_3 = 0

    if len(Loop1i_1_1) != 0:
        Velocity_Matrix[1, 0] = traci.vehicle.getSpeed(Loop1i_1_1[0])
        Loop1i_1_1 = 1
    else:
        Loop1i_1_1 = 0

    if len(Loop1i_1_2) != 0:
        Velocity_Matrix[1, 1] = traci.vehicle.getSpeed(Loop1i_1_2[0])
        Loop1i_1_2 = 1
    else:
        Loop1i_1_2 = 0

    if len(Loop1i_1_3) != 0:
        Velocity_Matrix[1, 2] = traci.vehicle.getSpeed(Loop1i_1_3[0])
        Loop1i_1_3 = 1
    else:
        Loop1i_1_3 = 0

    if len(Loop1i_2_1) != 0:
        Velocity_Matrix[2, 0] = traci.vehicle.getSpeed(Loop1i_2_1[0])
        Loop1i_2_1 = 1
    else:
        Loop1i_2_1 = 0

    if len(Loop1i_2_2) != 0:
        Velocity_Matrix[2, 1] = traci.vehicle.getSpeed(Loop1i_2_2[0])
        Loop1i_2_2 = 1
    else:
        Loop1i_2_2 = 0

    if len(Loop1i_2_3) != 0:
        Velocity_Matrix[2, 2] = traci.vehicle.getSpeed(Loop1i_2_3[0])
        Loop1i_2_3 = 1
    else:
        Loop1i_2_3 = 0

    if len(Loop1i_3_1) != 0:
        Velocity_Matrix[3, 0] = traci.vehicle.getSpeed(Loop1i_3_1[0])
        Loop1i_3_1 = 1
    else:
        Loop1i_3_1 = 0

    if len(Loop1i_3_2) != 0:
        Velocity_Matrix[3, 1] = traci.vehicle.getSpeed(Loop1i_3_2[0])
        Loop1i_3_2 = 1
    else:
        Loop1i_3_2 = 0

    if len(Loop1i_3_3) != 0:
        Velocity_Matrix[3, 2] = traci.vehicle.getSpeed(Loop1i_3_3[0])
        Loop1i_3_3 = 1
    else:
        Loop1i_3_3 = 0

    Position_Matrix[0, 0] = Loop1i_0_1
    Position_Matrix[0, 1] = Loop1i_0_2
    Position_Matrix[0, 2] = Loop1i_0_3
    Position_Matrix[1, 0] = Loop1i_1_1
    Position_Matrix[1, 1] = Loop1i_1_2
    Position_Matrix[1, 2] = Loop1i_1_3
    Position_Matrix[2, 0] = Loop1i_2_1
    Position_Matrix[2, 1] = Loop1i_2_2
    Position_Matrix[2, 2] = Loop1i_2_3
    Position_Matrix[3, 0] = Loop1i_3_1
    Position_Matrix[3, 1] = Loop1i_3_2
    Position_Matrix[3, 2] = Loop1i_3_3

    Loop2i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_1")
    Loop2i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_2")
    Loop2i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_3")
    Loop2i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_1")
    Loop2i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_2")
    Loop2i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_3")
    Loop2i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_1")
    Loop2i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_2")
    Loop2i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_3")
    Loop2i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_1")
    Loop2i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_2")
    Loop2i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_3")

    if len(Loop2i_0_1) != 0:
        Velocity_Matrix[4, 0] = traci.vehicle.getSpeed(Loop2i_0_1[0])
        Loop2i_0_1 = 1
    else:
        Loop2i_0_1 = 0

    if len(Loop2i_0_2) != 0:
        Velocity_Matrix[4, 1] = traci.vehicle.getSpeed(Loop2i_0_2[0])
        Loop2i_0_2 = 1
    else:
        Loop2i_0_2 = 0

    if len(Loop2i_0_3) != 0:
        Velocity_Matrix[4, 2] = traci.vehicle.getSpeed(Loop2i_0_3[0])
        Loop2i_0_3 = 1
    else:
        Loop2i_0_3 = 0

    if len(Loop2i_1_1) != 0:
        Velocity_Matrix[5, 0] = traci.vehicle.getSpeed(Loop2i_1_1[0])
        Loop2i_1_1 = 1
    else:
        Loop2i_1_1 = 0

    if len(Loop2i_1_2) != 0:
        Velocity_Matrix[5, 1] = traci.vehicle.getSpeed(Loop2i_1_2[0])
        Loop2i_1_2 = 1
    else:
        Loop2i_1_2 = 0

    if len(Loop2i_1_3) != 0:
        Velocity_Matrix[5, 2] = traci.vehicle.getSpeed(Loop2i_1_3[0])
        Loop2i_1_3 = 1
    else:
        Loop2i_1_3 = 0

    if len(Loop2i_2_1) != 0:
        Velocity_Matrix[6, 0] = traci.vehicle.getSpeed(Loop2i_2_1[0])
        Loop2i_2_1 = 1
    else:
        Loop2i_2_1 = 0

    if len(Loop2i_2_2) != 0:
        Velocity_Matrix[6, 1] = traci.vehicle.getSpeed(Loop2i_2_2[0])
        Loop2i_2_2 = 1
    else:
        Loop2i_2_2 = 0

    if len(Loop2i_2_3) != 0:
        Velocity_Matrix[6, 2] = traci.vehicle.getSpeed(Loop2i_2_3[0])
        Loop2i_2_3 = 1
    else:
        Loop2i_2_3 = 0

    if len(Loop2i_3_1) != 0:
        Velocity_Matrix[7, 0] = traci.vehicle.getSpeed(Loop2i_3_1[0])
        Loop2i_3_1 = 1
    else:
        Loop2i_3_1 = 0

    if len(Loop2i_3_2) != 0:
        Velocity_Matrix[7, 1] = traci.vehicle.getSpeed(Loop2i_3_2[0])
        Loop2i_3_2 = 1
    else:
        Loop2i_3_2 = 0

    if len(Loop2i_3_3) != 0:
        Velocity_Matrix[7, 2] = traci.vehicle.getSpeed(Loop2i_3_3[0])
        Loop2i_3_3 = 1
    else:
        Loop2i_3_3 = 0

    Position_Matrix[4, 0] = Loop2i_0_1
    Position_Matrix[4, 1] = Loop2i_0_2
    Position_Matrix[4, 2] = Loop2i_0_3
    Position_Matrix[5, 0] = Loop2i_1_1
    Position_Matrix[5, 1] = Loop2i_1_2
    Position_Matrix[5, 2] = Loop2i_1_3
    Position_Matrix[6, 0] = Loop2i_2_1
    Position_Matrix[6, 1] = Loop2i_2_2
    Position_Matrix[6, 2] = Loop2i_2_3
    Position_Matrix[7, 0] = Loop2i_3_1
    Position_Matrix[7, 1] = Loop2i_3_2
    Position_Matrix[7, 2] = Loop2i_3_3

    Loop3i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_1")
    Loop3i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_2")
    Loop3i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_3")
    Loop3i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_1")
    Loop3i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_2")
    Loop3i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_3")
    Loop3i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_1")
    Loop3i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_2")
    Loop3i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_3")
    Loop3i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_1")
    Loop3i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_2")
    Loop3i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_3")

    if len(Loop3i_0_1) != 0:
        Velocity_Matrix[8, 0] = traci.vehicle.getSpeed(Loop3i_0_1[0])
        Loop3i_0_1 = 1
    else:
        Loop3i_0_1 = 0

    if len(Loop3i_0_2) != 0:
        Velocity_Matrix[8, 1] = traci.vehicle.getSpeed(Loop3i_0_2[0])
        Loop3i_0_2 = 1
    else:
        Loop3i_0_2 = 0

    if len(Loop3i_0_3) != 0:
        Velocity_Matrix[8, 2] = traci.vehicle.getSpeed(Loop3i_0_3[0])
        Loop3i_0_3 = 1
    else:
        Loop3i_0_3 = 0

    if len(Loop3i_1_1) != 0:
        Velocity_Matrix[9, 0] = traci.vehicle.getSpeed(Loop3i_1_1[0])
        Loop3i_1_1 = 1
    else:
        Loop3i_1_1 = 0

    if len(Loop3i_1_2) != 0:
        Velocity_Matrix[9, 1] = traci.vehicle.getSpeed(Loop3i_1_2[0])
        Loop3i_1_2 = 1
    else:
        Loop3i_1_2 = 0

    if len(Loop3i_1_3) != 0:
        Velocity_Matrix[9, 2] = traci.vehicle.getSpeed(Loop3i_1_3[0])
        Loop3i_1_3 = 1
    else:
        Loop3i_1_3 = 0

    if len(Loop3i_2_1) != 0:
        Velocity_Matrix[10, 0] = traci.vehicle.getSpeed(Loop3i_2_1[0])
        Loop3i_2_1 = 1
    else:
        Loop3i_2_1 = 0

    if len(Loop3i_2_2) != 0:
        Velocity_Matrix[10, 1] = traci.vehicle.getSpeed(Loop3i_2_2[0])
        Loop3i_2_2 = 1
    else:
        Loop3i_2_2 = 0

    if len(Loop3i_2_3) != 0:
        Velocity_Matrix[10, 2] = traci.vehicle.getSpeed(Loop3i_2_3[0])
        Loop3i_2_3 = 1
    else:
        Loop3i_2_3 = 0

    if len(Loop3i_3_1) != 0:
        Velocity_Matrix[11, 0] = traci.vehicle.getSpeed(Loop3i_3_1[0])
        Loop3i_3_1 = 1
    else:
        Loop3i_3_1 = 0

    if len(Loop3i_3_2) != 0:
        Velocity_Matrix[11, 1] = traci.vehicle.getSpeed(Loop3i_3_2[0])
        Loop3i_3_2 = 1
    else:
        Loop3i_3_2 = 0

    if len(Loop3i_3_3) != 0:
        Velocity_Matrix[11, 2] = traci.vehicle.getSpeed(Loop3i_3_3[0])
        Loop3i_3_3 = 1
    else:
        Loop3i_3_3 = 0

    Position_Matrix[8, 0] = Loop3i_0_1
    Position_Matrix[8, 1] = Loop3i_0_2
    Position_Matrix[8, 2] = Loop3i_0_3
    Position_Matrix[9, 0] = Loop3i_1_1
    Position_Matrix[9, 1] = Loop3i_1_2
    Position_Matrix[9, 2] = Loop3i_1_3
    Position_Matrix[10, 0] = Loop3i_2_1
    Position_Matrix[10, 1] = Loop3i_2_2
    Position_Matrix[10, 2] = Loop3i_2_3
    Position_Matrix[11, 0] = Loop3i_3_1
    Position_Matrix[11, 1] = Loop3i_3_2
    Position_Matrix[11, 2] = Loop3i_3_3

    Loop4i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_1")
    Loop4i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_2")
    Loop4i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_3")
    Loop4i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_1")
    Loop4i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_2")
    Loop4i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_3")
    Loop4i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_1")
    Loop4i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_2")
    Loop4i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_3")
    Loop4i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_1")
    Loop4i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_2")
    Loop4i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_3")

    if len(Loop4i_0_1) != 0:
        Velocity_Matrix[12, 0] = traci.vehicle.getSpeed(Loop4i_0_1[0])
        Loop4i_0_1 = 1
    else:
        Loop4i_0_1 = 0

    if len(Loop4i_0_2) != 0:
        Velocity_Matrix[12, 1] = traci.vehicle.getSpeed(Loop4i_0_2[0])
        Loop4i_0_2 = 1
    else:
        Loop4i_0_2 = 0

    if len(Loop4i_0_3) != 0:
        Velocity_Matrix[12, 2] = traci.vehicle.getSpeed(Loop4i_0_3[0])
        Loop4i_0_3 = 1
    else:
        Loop4i_0_3 = 0

    if len(Loop4i_1_1) != 0:
        Velocity_Matrix[13, 0] = traci.vehicle.getSpeed(Loop4i_1_1[0])
        Loop4i_1_1 = 1
    else:
        Loop4i_1_1 = 0

    if len(Loop4i_1_2) != 0:
        Velocity_Matrix[13, 1] = traci.vehicle.getSpeed(Loop4i_1_2[0])
        Loop4i_1_2 = 1
    else:
        Loop4i_1_2 = 0

    if len(Loop4i_1_3) != 0:
        Velocity_Matrix[13, 2] = traci.vehicle.getSpeed(Loop4i_1_3[0])
        Loop4i_1_3 = 1
    else:
        Loop4i_1_3 = 0

    if len(Loop4i_2_1) != 0:
        Velocity_Matrix[14, 0] = traci.vehicle.getSpeed(Loop4i_2_1[0])
        Loop4i_2_1 = 1
    else:
        Loop4i_2_1 = 0

    if len(Loop4i_2_2) != 0:
        Velocity_Matrix[14, 1] = traci.vehicle.getSpeed(Loop4i_2_2[0])
        Loop4i_2_2 = 1
    else:
        Loop4i_2_2 = 0

    if len(Loop4i_2_3) != 0:
        Velocity_Matrix[14, 2] = traci.vehicle.getSpeed(Loop4i_2_3[0])
        Loop4i_2_3 = 1
    else:
        Loop4i_2_3 = 0

    if len(Loop4i_3_1) != 0:
        Velocity_Matrix[15, 0] = traci.vehicle.getSpeed(Loop4i_3_1[0])
        Loop4i_3_1 = 1
    else:
        Loop4i_3_1 = 0

    if len(Loop4i_3_2) != 0:
        Velocity_Matrix[15, 1] = traci.vehicle.getSpeed(Loop4i_3_2[0])
        Loop4i_3_2 = 1
    else:
        Loop4i_3_2 = 0

    if len(Loop4i_3_3) != 0:
        Velocity_Matrix[15, 2] = traci.vehicle.getSpeed(Loop4i_3_3[0])
        Loop4i_3_3 = 1
    else:
        Loop4i_3_3 = 0

    Position_Matrix[12, 0] = Loop4i_0_1
    Position_Matrix[12, 1] = Loop4i_0_2
    Position_Matrix[12, 2] = Loop4i_0_3
    Position_Matrix[13, 0] = Loop4i_1_1
    Position_Matrix[13, 1] = Loop4i_1_2
    Position_Matrix[13, 2] = Loop4i_1_3
    Position_Matrix[14, 0] = Loop4i_2_1
    Position_Matrix[14, 1] = Loop4i_2_2
    Position_Matrix[14, 2] = Loop4i_2_3
    Position_Matrix[15, 0] = Loop4i_3_1
    Position_Matrix[15, 1] = Loop4i_3_2
    Position_Matrix[15, 2] = Loop4i_3_3

    # Create 4 x 1 matrix for phase state
    Phase = []
    if traci.trafficlight.getPhase('JO') == 0 or traci.trafficlight.getPhase('J0') == 1 or traci.trafficlight.getPhase(
            'J0') == 2 or traci.trafficlight.getPhase('J0') == 3:
        Phase = [1, 0, 0, 0]
    elif traci.trafficlight.getPhase('J0') == 4 or traci.trafficlight.getPhase('J0') == 5 or traci.trafficlight.getPhase(
            'J0') == 6 or traci.trafficlight.getPhase('J0') == 7:
        Phase = [0, 1, 0, 0]
    elif traci.trafficlight.getPhase('J0') == 8 or traci.trafficlight.getPhase('J0') == 9 or traci.trafficlight.getPhase(
            'J0') == 10 or traci.trafficlight.getPhase('J0') == 11:
        Phase = [0, 0, 1, 0]
    elif traci.trafficlight.getPhase('J0') == 12 or traci.trafficlight.getPhase('J0') == 13 or traci.trafficlight.getPhase(
            'J0') == 14 or traci.trafficlight.getPhase('J0') == 15:
        Phase = [0, 0, 0, 1]

    Phase = np.array(Phase)
    Phase = Phase.flatten()

    # state = np.concatenate((Position_Matrix, Velocity_Matrix), axis=0)
    # state = state.flatten()
    # state = np.concatenate((state, Phase), axis=0)
    state = np.concatenate((Position_Matrix.flatten(), Velocity_Matrix.flatten()))
    # Create matrix for duration
    Duration_Matrix = [traci.trafficlight.getPhaseDuration('0')]

    Duration_Matrix = np.array(Duration_Matrix)
    Duration_Matrix = Duration_Matrix.flatten()
    state = np.concatenate((state, Duration_Matrix.flatten()))

    return state