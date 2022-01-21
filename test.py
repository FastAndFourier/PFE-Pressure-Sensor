import pybullet as p
import pybullet_data
from time import sleep

def main():
    PYB_CLIENT = p.connect(p.GUI, options = "--opengl2")
    p.configureDebugVisualizer()
    #p.resetDebugVisualizerCamera()
    p.setGravity(0, 0, -9.8, physicsClientId=PYB_CLIENT)
    p.setRealTimeSimulation(0, physicsClientId=PYB_CLIENT)
    p.setTimeStep(1/240, physicsClientId=PYB_CLIENT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=PYB_CLIENT)
    plane = p.loadURDF("plane.urdf", physicsClientId=PYB_CLIENT)
    duck = p.loadURDF("leg_spot_center.urdf", [0, 0, 10], physicsClientId=PYB_CLIENT)
    for _ in range(240*10):
        p.stepSimulation(physicsClientId=PYB_CLIENT)
        sleep(1/(240*10))
        if(p.getClosestPoints(bodyA =duck, bodyB = plane, distance = 0.1)!=()):
            print( p.getClosestPoints(bodyA =duck, bodyB = plane, distance = 0.1))
            #print("second elem", p.getContactPoints(bodyA =duck, bodyB = plane)[1], "\n")
            
            #input()

if __name__ == "__main__":
    main()