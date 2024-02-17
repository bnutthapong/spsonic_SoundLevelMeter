
import serial
import time
import os
import sys


sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/home/pi/SPsonic')
sys.path.append('./.local/lib/python3.7/site-packages')


from pymodbus.client.sync import ModbusSerialClient
client = ModbusSerialClient(method = "rtu", port="/dev/ttyUSB0",stopvits = 1, bytesize = 8, parity ="N", baudrate = 9600)


ser=serial.Serial("/dev/ttyS0",baudrate = 115200,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS,timeout=1)
os.system('clear')

Check_conn = client.connect()
if Check_conn == True:
  print(">>>> Connected to PLC: ", Check_conn)
else:
  client.close()
  ser.close()

print("---- Program starting ----")
print("If you want to exit, please press Ctrl+c")

t=0
x=0
SPL = 0
c=''

while 1:
    Check_conn = client.connect()
    if Check_conn == False:
      break
    #print result
    result = client.read_input_registers(0x00,1, unit=0x01)
    t = result.registers[0]
    #print(t)

    #print("SPL:", t/10)
    SPL = t/10

    #print(SPL)
    #client.close()


    if SPL < 10:
        x = "S00" + str(SPL) + "\r\n"
    elif SPL > 100 :
        x = "S" + str(SPL) + "\r\n"
    else:
        x = "S0" + str(SPL) + "\r\n"

    a = ser.write([ord(c) for c in x])
    ser.flush()
    tt = time.localtime()
    current_time = time.strftime("%H:%M:%S",tt)
    #print(current_time+":"+str(x))
    #time.sleep(0.015)
    #time.sleep(0.00005)


