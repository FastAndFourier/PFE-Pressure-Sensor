import pybullet as p
import pybullet_data

import time

import cv2

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":

    # Connect physical server
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    p.enableCollision()


    # Plane loading
    planeId = p.loadURDF("plane.urdf")

    # # Creating pressure sensor as a MultiBody + grounding it to the plane
    # visualSensor = p.createVisualShape(shapeType=p.GEOM_BOX,
    #                        halfExtents=[0.5,0.5,0.05],
    #                        rgbaColor=[0.321, 0.827, 0.898, 1])

    # collisionSensor = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=[0.5,0.5,0.05])

    # sensor = p.createMultiBody(baseMass=1000,
    #                            baseCollisionShapeIndex=collisionSensor,
    #                            baseVisualShapeIndex=visualSensor,
    #                            basePosition=[0, 0, 0.05],
    #                            baseInertialFramePosition=[0,0,0.05])

    # p.createConstraint(parentBodyUniqueId=planeId,parentLinkIndex=-1,
    #                    childBodyUniqueId=sensor,childLinkIndex=-1,
    #                    jointType=p.JOINT_FIXED,jointAxis=[0,0,0],
    #                    parentFramePosition=[0,0,0],childFramePosition=[0,0,0])

    # sphere = vtk.vtkSphereSource()
    # sphere.SetRadius(4)
    # sphere.SetCenter(0,0,1)
    # sphere.Update()
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetInputData(sphere.GetOutput())
    # writer.SetFileName('mysphere.vtk')
    # writer.Update()

    # Loading leg model
    leg = p.loadURDF("leg.urdf",useFixedBase=True)

    min_val_joint = []
    max_val_joint = []

    

    for joint_id in range(p.getNumJoints(leg)):

        max_val_joint.append(p.getJointInfo(leg,joint_id)[9])
        min_val_joint.append(p.getJointInfo(leg,joint_id)[8])

        p.resetJointState(leg,joint_id,0)#max_val_joint[-1])
        p.setJointMotorControl2(leg, joint_id, p.POSITION_CONTROL, 0,0)#max_val_joint[-1], 0)

    n_iter = 5000

    angle1 = p.addUserDebugParameter('upper', 0, 1, 0)
    angle2 = p.addUserDebugParameter('middle', -1, 1, 0)
    angle3 = p.addUserDebugParameter('lower', -1, 1, 0)

    contact_point=[]
    ray=[]

    for it in range(n_iter):
        
        angle=[p.readUserDebugParameter(angle1),p.readUserDebugParameter(angle2),p.readUserDebugParameter(angle3),0]

        for id_joint in range(0,p.getNumJoints(leg),2):
            pos = angle[id_joint//2]
            #pos = min_val_joint[id_joint] + (max_val_joint[id_joint] - min_val_joint[id_joint])*it/n_iter
            p.setJointMotorControl2(bodyIndex=leg, jointIndex=id_joint, controlMode=p.POSITION_CONTROL, targetPosition=pos)

        p.performCollisionDetection()

        dist = 1e-3
        nb_closest_point = len(p.getClosestPoints(bodyA=leg,bodyB=planeId,distance=dist,linkIndexA=p.getNumJoints(leg)-1))
        contact_point.append(nb_closest_point)

        if nb_closest_point!=0:
            print(nb_closest_point)


        p.stepSimulation()
        time.sleep(1./60.)

    plt.plot(contact_point)
    #plt.imshow(ray[0])
    plt.show()