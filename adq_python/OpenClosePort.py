import serial # pip install pyserial

# Open the serial port
ser = serial.Serial('COM5',
                    6000000,
                    timeout=None,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    rtscts=False)
print('rts', ser.rts)
print('dtr', ser.dtr)
print('is open', ser.is_open)
ser.rts = False
ser.dtr = False
# print('is open', ser.is_open)
if not ser.is_open:
    print('Opening port...')
    ser.open()
print('is open', ser.is_open)
ser.rtscts = True
ser.close()
print('is open', ser.is_open)