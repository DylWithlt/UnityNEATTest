using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgentsExamples;
using UnitySharpNEAT;
using BodyPart = Unity.MLAgentsExamples.BodyPart;
using SharpNeat.Phenomes;
using System;

public class VectorSensor {
    public List<double> Observations = new List<double>();

    public void AddObservation(bool observation) {
        Observations.Add(observation ? 1 : 0);
    }

    public void AddObservation(float observation) {
        Observations.Add(observation);
    }

    public void AddObservation(Vector3 observation) {
        Observations.Add(observation.x);
        Observations.Add(observation.y);
        Observations.Add(observation.z);
    }

    public void AddObservation(Quaternion observation) {
        Observations.Add(observation.x);
        Observations.Add(observation.y);
        Observations.Add(observation.z);
        Observations.Add(observation.w);
    }
}

[RequireComponent(typeof(JointDriveController))]
public class WalkingAgent : UnitController
{
    private Vector3 Mask2d = new Vector3(1, 0, 1);
    public float timeSinceStart = 0f;
    public float onGroundScore = 0f;

    private Vector3 lastHipsPosition;
    private float totalDistanceTraveled;
    private float avgSpeed;
    private Vector3 avgVel; 

    [Header("Body Parts")]
    public Transform hips;
    //public Transform waist;
    //public Transform body;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;

    [Header("Fitness")]
    public float distanceMultiplier = 1.2f;
    public float avgSpeedMultiplier = 0.2f;
    public float stayAliveMultiplier = 0.2f;
    public float onGroundMultiplier = 0.6f;

    [Header("Info View")]
    public float overallFitness = 0;

    JointDriveController m_JdController;
    OrientationCubeController m_OrientationCube;

    // Take inputs for the BlackBox to feed into the NEAT algorithm.
    protected override void UpdateBlackBoxInputs(ISignalArray inputSignalArray)
    {
        VectorSensor sensor = new VectorSensor();

        InputSensors(sensor);
        double[] observations = sensor.Observations.ToArray();
        inputSignalArray.CopyFrom(observations, 0, observations.Length);

        timeSinceStart += Time.deltaTime;

        var bpDict = m_JdController.bodyPartsDict;
        if (bpDict[footL].groundContact || bpDict[footR].groundContact && onGroundScore < 200)
        {
            onGroundScore += 1;
        }
        
        lastHipsPosition = hips.transform.position;
    }

    // Use the output from the NEAT neural net to move the body.
    protected override void UseBlackBoxOutpts(ISignalArray outputSignalArray)
    {
        MoveBody(outputSignalArray);
    }

    // Calculate the fitness  based on totalDistanceTravelled, how much they remain on the ground, and timeSinceStart.
    public override float GetFitness()
    {
        totalDistanceTraveled = Vector3.Distance(Vector3.Scale(Mask2d, m_JdController.bodyPartsDict[hips].startingPos), Vector3.Scale(Mask2d, hips.position));
        //avgSpeed = totalDistanceTraveled / timeSinceStart;
        Debug.Log(totalDistanceTraveled);
        //Debug.Log(avgSpeed);
        return (totalDistanceTraveled * distanceMultiplier) + (onGroundScore * onGroundMultiplier) + (timeSinceStart * stayAliveMultiplier);
    }

    // When the NEAT system calls this method the rig is reset or if its set off the rig turns off it's components hiding the rig.
    protected override void HandleIsActiveChanged(bool newIsActive)
    {
        if (newIsActive == true)
        {
            Reset();
        }

        foreach (Transform t in transform)
        {
            t.gameObject.SetActive(newIsActive);
        }
    }

    // Method called by the touch detector scripts inside each bodypart.
    public void Death()
    {
        IsActive = false;
    }

    // When the rig starts we want to setup its bodyparts.
    private void Awake() {
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();

        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        //m_JdController.SetupBodyPart(body);
        //m_JdController.SetupBodyPart(waist);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);
        m_JdController.SetupBodyPart(footR);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(footL);
    }

    // This will restore all variables to default values and put bodyparts in original positions for a new run.
    public void Reset() {
        timeSinceStart = 0f;
        totalDistanceTraveled = 0f;
        avgSpeed = 0f;
        avgVel = Vector3.zero;
        onGroundScore = 0f;

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values) {
            bodyPart.Reset(bodyPart);
        }
    }

    // Currently used only for the neural net input.
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.velocity;
        }

        var avgVel = velSum / numOfRb;
        return avgVel;
    }

    // Uses the sensor to record observations about groundContact, orientation cube similarity, and rotation + joint strength scaled for a bodypart. 
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor) {

        sensor.AddObservation(bp.groundContact.touchingGround);

        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));

        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));

        if (bp.rb.transform == hips) { return; }

        sensor.AddObservation(bp.rb.transform.localRotation);
        sensor.AddObservation(bp.currentStrength/m_JdController.maxJointForceLimit);
    }

    // Gets all the input for the Neural Net.
    private void InputSensors(VectorSensor sensor) {
        avgVel = GetAvgVelocity();

        sensor.AddObservation(avgVel);

        foreach (var bodyPart in m_JdController.bodyPartsList) {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    // Sets the joints target and strength for all the joints. 
    public void MoveBody(ISignalArray continuousActions) {
        var bpDict = m_JdController.bodyPartsDict;
        var i = -1;

        //bpDict[body].SetJointTargetRotation((float)continuousActions[++i], (float)continuousActions[++i], 0);
        //bpDict[waist].SetJointTargetRotation((float)continuousActions[++i], (float)continuousActions[++i], 0);

        bpDict[thighR].SetJointTargetRotation((float)continuousActions[++i], (float)continuousActions[++i], 0);
        bpDict[shinR].SetJointTargetRotation((float)continuousActions[++i], 0, 0);
        bpDict[footR].SetJointTargetRotation((float)continuousActions[++i], (float)continuousActions[++i], (float)continuousActions[++i]);
        bpDict[thighL].SetJointTargetRotation((float)continuousActions[++i], (float)continuousActions[++i], 0);
        bpDict[shinL].SetJointTargetRotation((float)continuousActions[++i], 0, 0);
        bpDict[footL].SetJointTargetRotation((float)continuousActions[++i], (float)continuousActions[++i], (float)continuousActions[++i]);

        //bpDict[body].SetJointStrength((float)continuousActions[++i]);
        //bpDict[waist].SetJointStrength((float)continuousActions[++i]);
        bpDict[thighR].SetJointStrength((float)continuousActions[++i]);
        bpDict[shinR].SetJointStrength((float)continuousActions[++i]);
        bpDict[footR].SetJointStrength((float)continuousActions[++i]);
        bpDict[thighL].SetJointStrength((float)continuousActions[++i]);
        bpDict[shinL].SetJointStrength((float)continuousActions[++i]);
        bpDict[footL].SetJointStrength((float)continuousActions[++i]);
    }
}
