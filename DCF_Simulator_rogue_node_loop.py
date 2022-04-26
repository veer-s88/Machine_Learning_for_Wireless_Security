"""
This file is the main simulation for a wireless network using DCF in 802.11a
Multiple parameters to edit: number of nodes, total contention rounds
Results for the simulation are printed on the console at the end
"""
import numpy as np
import math

import pandas as pd


def dcf_sim():
    # initialise variables
    no_of_nodes_lst = []
    collision_probability_lst = []
    total_throughput_lst = []
    avg_delay_lst = []
    sim_time_lst = []
    dropped_packet_lst = []

    # m = Number of nodes (competing stations)
    for m in range(1, 61):
        total_contention_rounds = 100000
        rnd = 0
        cw_min = 15
        cw_max = 1023
        retry_limit = 7

        # initialise data and control rate variables
        data_rate = 54  # Mbps
        control_rate = 24  # Mbps

        # initialise arrays to store information
        collisions = np.zeros(m)
        successful_transmissions = np.zeros(m)
        retransmissions = np.zeros(m)
        cw_per_node = np.ones(m) * cw_min
        retry_limit_cnt = np.zeros(m)
        payload_bytes_transmitted = np.zeros(m)
        tx_time = np.zeros(total_contention_rounds)

        # initial random backoff timer for all nodes within contention window
        curr_back_off = np.random.randint(0, cw_min+1, m)
        next_back_off = np.zeros(m)

        while rnd < total_contention_rounds:
            # array of nodes which have smallest back off timer, i.e., are next to transmit
            tx_nodes = np.where(curr_back_off == np.min(curr_back_off))[0]

            # set random payload value in bytes
            mac_payload = np.random.randint(2312)
            collision = False

            """
            # -- debug info
            print('current back off timers = ', curr_back_off)
            print('index of tx nodes = ', tx_nodes)
            """

            # if more than 1 transmitting node, assume collision
            if len(tx_nodes) > 1:
                collision = True
                collisions[tx_nodes] += 1
                retransmissions[tx_nodes] += 1
                next_back_off = curr_back_off - np.min(curr_back_off)

                for node in tx_nodes:
                    if retransmissions[node] > retry_limit:
                        retry_limit_cnt[node] += 1
                        cw_per_node[node] = cw_min
                        retransmissions[node] = 0
                    # rogue Wi-Fi node does not adhere to the binary exponential algorithm
                    elif node >= (0.5*m - 1):
                        cw_per_node[node] = cw_min
                    else:
                        cw_per_node[node] = min(cw_per_node[node]*2, cw_max)
                    next_back_off[node] = np.random.randint(0, cw_per_node[node] + 1)

            # else if only one transmitting node, assume successful transmission
            else:
                successful_transmissions[tx_nodes] += 1
                retransmissions[tx_nodes] = 0
                payload_bytes_transmitted[tx_nodes] += mac_payload
                next_back_off = curr_back_off - np.min(curr_back_off)
                cw_per_node[tx_nodes] = cw_min
                next_back_off[tx_nodes] = np.random.randint(0, cw_min+1, len(tx_nodes))

            # transmission time of each contention round whether collision or not including backoff times
            tx_time[rnd] = transmission_time(np.min(curr_back_off), data_rate, control_rate, mac_payload, collision)

            rnd += 1
            curr_back_off = next_back_off

            """
            # -- debug info, prints arrays for each contention round
            print('collisions per node = ', collisions)
            print('successful transmissions per node = ', successful_transmissions)
            print('contention window per node = ', cw_per_node)
            print('next back off per node =', next_back_off)
            print('MAC payload bytes = ', mac_payload)
            print('transmission time per packet = ', tx_time)
            print('')
            """

        # calculate throughput, simulation time, average delay per packet
        simulation_time = np.sum(tx_time) * 9e-6
        throughput_per_node = payload_bytes_transmitted * 8 / simulation_time
        total_network_throughput = np.sum(throughput_per_node)
        avg_delay = simulation_time / successful_transmissions

        # print results of simulation
        print('collisions per node = ', collisions)
        print('successful transmissions per node = ', successful_transmissions)
        print('total_collisions =', total_contention_rounds - sum(successful_transmissions))
        print('total successful transmissions = ', sum(successful_transmissions))
        print('Number of times retry limit reached per node = ', retry_limit_cnt)
        print('Total number of times retry limit is reached = ', sum(retry_limit_cnt))
        print('Total number of attempted packets sent = ', sum(successful_transmissions)+sum(collisions))
        print('collision_prob_per_packet = ', sum(collisions)/(sum(successful_transmissions)+sum(collisions)))
        print('collision_prob_per_contention_round = ', (total_contention_rounds-sum(successful_transmissions)) /
              total_contention_rounds)
        print('total_simulation_time = ', simulation_time)
        print('throughput per node = ', throughput_per_node)
        print('Total network throughput (b/s) = ', total_network_throughput)
        print('Payload bytes transmitted = ', payload_bytes_transmitted)
        print('Average delay per packet per node = ', avg_delay)
        print('Average delay per packet for all nodes = ', np.mean(avg_delay))
        # print('Transmission time per packet = ', tx_time)

        # append values to individual lists
        no_of_nodes_lst.append(m)
        collision_probability_lst.append(sum(collisions) / (sum(successful_transmissions) + sum(collisions)))
        total_throughput_lst.append(total_network_throughput)
        avg_delay_lst.append(np.mean(avg_delay))
        sim_time_lst.append(simulation_time)
        dropped_packet_lst.append(sum(retry_limit_cnt))

        print('')

    # Create dictionary and dataframe of results
    dictionary = {'number of nodes': no_of_nodes_lst,
                  'collision probability': collision_probability_lst,
                  'total throughput': total_throughput_lst,
                  'average delay': avg_delay_lst,
                  'simulation time': sim_time_lst,
                  'dropped packet': dropped_packet_lst
                  }
    df = pd.DataFrame(dictionary)
    # Export to csv file
    df.to_csv('simulation_results_rogue_node_50_perc.csv')
    print(df)


def transmission_time(backoff_slots, data_rate, control_rate, mac_payload, collision):

    coded_bits_per_symbol = {6: 48, 9: 48, 12: 96, 18: 96, 24: 192, 36: 192, 48: 288, 54: 288}

    slot_time = 9e-6  # seconds
    sifs = 16e-6  # seconds
    difs = sifs + 2*slot_time

    plcp_preamble = 16e-6
    plcp_signal = 4e-6

    service = 16  # bits
    tail = 6  # bits

    mac_header = 30  # bytes
    mac_tail = 4  # bytes
    mac_frame = (mac_header + mac_payload + mac_tail)*8  # bits

    n_sym = math.ceil((service + 8 * (mac_header + mac_payload + mac_tail) + tail) / coded_bits_per_symbol[data_rate])
    n_data = n_sym * coded_bits_per_symbol[data_rate]
    n_pad = n_data - (service + 8 * (mac_header + mac_payload + mac_tail) + tail)  # bits

    data_duration = plcp_preamble + plcp_signal + (service + mac_frame + tail + n_pad)/(data_rate * 1e6)

    ack = 14 * 8  # bits
    ack_duration = plcp_preamble + plcp_signal + (service + ack + tail) / (control_rate * 1e6)

    tx_time = difs + data_duration  # seconds

    if not collision:
        tx_time += sifs + ack_duration  # seconds
    else:
        # simulating ack timeout under collision
        tx_time += 2*sifs + ack_duration  # seconds

    tx_time_slots = math.ceil(tx_time/slot_time) + backoff_slots

    return tx_time_slots


dcf_sim()
