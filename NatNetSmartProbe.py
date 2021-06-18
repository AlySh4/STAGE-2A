import csv
from OriginalTemplates.NatNetClient import NatNetClient
from time import sleep
from collections import deque
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-sp', action='append', nargs=2,
                    metavar=('rigid_id', 'ac_id'), help='pair of rigid body and A/C id (multiple possible)')
parser.add_argument('-s', '--server', dest='server', default="127.0.0.1", help="NatNet server IP address")
parser.add_argument('-m', '--multicast_addr', dest='multicast', default="239.255.42.99",
                    help="NatNet server multicast address")
parser.add_argument('-dp', '--data_port', dest='data_port', type=int, default=1511,
                    help="NatNet server data socket UDP port")
parser.add_argument('-cp', '--command_port', dest='command_port', type=int, default=1510,
                    help="NatNet server command socket UDP port")
parser.add_argument('-f', '--freq', dest='freq', default=10, type=int, help="transmit frequency")
parser.add_argument('-gr', '--ground_ref', dest='ground_ref', action='store_true',
                    help="also send the GROUND_REF message")
parser.add_argument('-vs', '--vel_samples', dest='vel_samples', default=4, type=int,
                    help="amount of samples to compute velocity (should be greater than 2)")
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="display debug messages")
args = parser.parse_args()

if args.sp is None:
    print("Please declare the pair of rigid boby / smartprobe id")
    exit()

id_dict = dict(args.sp)

timestamp = dict([(ac_id, None) for ac_id in id_dict.keys()])
period = 1. / args.freq

# initial track per AC
track = dict([(ac_id, deque()) for ac_id in id_dict.keys()])


def storeOptiTrackData1():
    while True:
        with open('Data/OptiTrackData.csv', 'w', newline='') as f:
            dataWriter = csv.writer(f, escapechar=' ', quoting=csv.QUOTE_NONE)
            a = list(natnet.rigidBodyList[0])
            b = []
            for element in a:
                if type(element) is tuple:
                    for i in element:
                        b.append(i)
                else:
                    b.append(element)
            dataWriter.writerow(b)
            f.flush()
            sleep(period)


def storeOptiTrackData():
    with open('Data/OptiTrackData.csv', 'w', newline='') as f:
        dataWriter = csv.writer(f, escapechar=' ', quoting=csv.QUOTE_NONE)
        while True:
            a = list(natnet.rigidBodyList[0])
            b = []
            for element in a:
                if type(element) is not bool:
                    if type(element) is tuple:
                        for i in element:
                            b.append(i)
                    else:
                        b.append(element)

            dataWriter.writerow(b)
            f.flush()
            sleep(period)


def store_track(ac_id, pos, t):
    if ac_id in id_dict.keys():
        track[ac_id].append((pos, t))
        if len(track[ac_id]) > args.vel_samples:
            track[ac_id].popleft()


def receiveRigidBodyList(rigidBodyList, stamp):
    for (ac_id, pos, quat, valid) in rigidBodyList:
        if not valid:
            # skip if rigid body is not valid
            continue

        i = str(ac_id)
        if i not in id_dict.keys():
            continue

        store_track(i, pos, stamp)
        if timestamp[i] is None or abs(stamp - timestamp[i]) < period:
            if timestamp[i] is None:
                timestamp[i] = stamp
            continue  # too early for next message
        timestamp[i] = stamp


natnet = NatNetClient(
    server=args.server,
    rigidBodyListListener=receiveRigidBodyList,
    dataPort=args.data_port,
    commandPort=args.command_port,
    verbose=args.verbose)

print("Starting Natnet3.x to Ivy interface at %s" % (args.server))
try:
    # Start up the streaming client.
    # This will run perpetually, and operate on a separate thread.
    natnet.run()
    sleep(0.1)
    storeOptiTrackData1()
except (KeyboardInterrupt, SystemExit):
    print("Shutting down natnet interfaces...")
    natnet.stop()
except OSError:
    print("Natnet connection error")
    natnet.stop()
    exit(-1)
