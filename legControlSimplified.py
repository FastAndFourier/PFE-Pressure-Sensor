import pybullet as p
import pybullet_data
import time

import matplotlib.pyplot as plt

  
if __name__ == "__main__":

    # Connect physical server
    client = p.connect(p.GUI, options="--opengl2")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)
    
    # Plane loading
    planeId = p.loadURDF("plane.urdf", physicsClientId=client)


    # # Creating pressure sensor as a MultiBody + grounding it to the plane
    # visualSensor = p.createVisualShape(shapeType=p.GEOM_BOX,
    #                        halfExtents=[0.5,0.5,0.1],
    #                        rgbaColor=[0.321, 0.827, 0.898, 1])

    # collisionSensor = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=[0.5,0.5,0.1])

    # sensor = p.createMultiBody(baseMass=1000,
    #                            baseCollisionShapeIndex=collisionSensor,
    #                            baseVisualShapeIndex=visualSensor,
    #                            basePosition=[0, 0, 0.1],
    #                            baseInertialFramePosition=[0,0,0.1])

    # p.createConstraint(parentBodyUniqueId=planeId,parentLinkIndex=-1,
    #                    childBodyUniqueId=sensor,childLinkIndex=-1,
    #                    jointType=p.JOINT_FIXED,jointAxis=[0,0,0],
    #                    parentFramePosition=[0,0,0],childFramePosition=[0,0,0.1])

    # Loading leg model
    leg = p.loadURDF("leg_spot_center.urdf", useFixedBase=True, physicsClientId=client)

    min_val_joint = []
    max_val_joint = []


    #print(p.getLinkState(leg, 1))
    p.performCollisionDetection()

    n_iter = 10000

    angle = p.addUserDebugParameter('upper', -3.14, 3.14, 0)


    for it in range(n_iter):
        
        pos=p.readUserDebugParameter(angle)
        p.setJointMotorControl2(bodyIndex=leg, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=pos)
        
        p.stepSimulation()
        time.sleep(1./240.)

        if(p.getContactPoints(bodyA =leg, bodyB = planeId)!=()):
            print("first elem", p.getContactPoints(bodyA = leg, bodyB = planeId)[0], "\n")
            print("second elem", p.getContactPoints(bodyA = leg, bodyB = planeId)[1], "\n")
            #input()