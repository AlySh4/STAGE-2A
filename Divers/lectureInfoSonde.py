import pprzlink.serial
import pprzlink.messages_xml_map as messages_xml_map
import pprzlink.message as message
import time
import csv

import argparse


class SerialTutorial:
    """
    Class SerialTutorial that uses pprzlink.serial.SerialMessagesInterface to send
    PING messages on a serial device and monitors incoming messages.
    It respond to PING messages with PONG messages.
    """

    # Construction of the SerialTutorial object
    def __init__(self, args):
        self.serial_interface = pprzlink.serial.SerialMessagesInterface(
            callback=self.storeSmartProbeData,  # callback function
            device=args.dev,  # serial device
            baudrate=args.baud,  # baudrate
            # interface_id=args.ac_id,  # id of the aircraft
        )
        # self.ac_id = args.ac_id
        self.baudrate = args.baud

        # Main loop of the tutorial

    def run(self):
        print("Starting serial interface on %s at %i baud" % (args.dev, self.baudrate))

        try:
            self.serial_interface.start()

            # give the thread some time to properly start
            time.sleep(0.1)

            while self.serial_interface.isAlive():
                self.serial_interface.join(1)

        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            self.serial_interface.stop()
            exit()

    # Callback function that process incoming messages
    def process_incoming_message(self, source, pprz_message):
        print("Received message from %i: %s" % (source, pprz_message))

    def storeSmartProbeData(self, source, pprz_message):
        a = str(pprz_message)[20:]
        print(a)
        a = a.split(',')
        TAS = float(a[10].split(' ')[-1])
        EAS = float(a[11].split(' ')[-1])
        alpha = float(a[12].split(' ')[-1])
        beta = float(a[13].split(' ')[-1])
        with open('../Data/SmartProbeData.csv', 'w', newline='') as f:
            dataWriter = csv.writer(f)
            dataWriter.writerow([TAS, EAS, alpha, beta])
            f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="device name", dest='dev', default='COM7')
    parser.add_argument("-b", "--baudrate", help="baudrate", dest='baud', default=230400, type=int)
    # parser.add_argument("-id", "--ac_id", help="aircraft id (receiver)", dest='ac_id', default=42, type=int)
    args = parser.parse_args()

    serialTutorial = SerialTutorial(args)

    serialTutorial.run()
