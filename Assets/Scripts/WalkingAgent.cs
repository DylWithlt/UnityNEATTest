using SharpNeat.Phenomes;
using System;
using System.Collections.Generic;
using Unity.MLAgentsExamples;
using UnityEngine;
using UnitySharpNEAT;
using BodyPart = Unity.MLAgentsExamples.BodyPart;

public class VectorSensor
{
    public List<double> Observations = new List<double>();

    public void AddObservation(bool observation)
    {
        Observations.Add(observation ? 1 : 0);
    }

    public void AddObservation(float observation)
    {
        Observations.Add(observation);
    }

    public void AddObservation(Vector3 observation)
    {
        Observations.Add(observation.x);
        Observations.Add(observation.y);
        Observations.Add(observation.z);
    }

    public void AddObservation(Quaternion observation)
    {
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

    [Header("Control Variables")]
    public float targetFootZDistance = 5f;
    public float penaltyUnderY = 0.5f;

    [Header("Agent Variable Tracking")]
    public float totalDistanceTraveled = 0;
    public float timeSinceStart = 0f;
    public float onGroundScore = 0f;
    public float fallingOverPentalty = 0f;
    public float worstStandingZ = 0f;
    public float stepByStepScore = 0f;
    public float veeringPenalty = 0f;
    public float timeSpentMoving = 0f;

    public List<double> lastFootDistances = new List<double>();

    private Vector3 avgVel = Vector3.zero;
    private float lastCoMZ = 0f;
    private Vector3 CoM = Vector3.zero;

    public Transform lastBestGroundedFoot;
    public float lastBestGroundedFootZ = 0f;

    [Header("Fitness Multipliers")]
    public float distanceMultiplier = 2f;
    public float avgSpeedMultiplier = 0.2f;
    public float stayAliveMultiplier = 0.2f;
    public float onGroundMultiplier = 2f;
    public float fallingOverPenaltyMultiplier = 0.5f;
    public float stepByStepMultiplier = 3f;
    public float veeringPenaltyMultiplier = 1f;

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

    JointDriveController m_JdController;
    OrientationCubeController m_OrientationCube;

    private void updateStepByStep(float groundedFootZ)
    {
        float footZDisplacement = (lastBestGroundedFootZ - groundedFootZ);
        float distanceFromTarget = Mathf.Abs(targetFootZDistance - footZDisplacement);
        Debug.Log(distanceFromTarget);
        if (distanceFromTarget <= 3f)
        {
            stepByStepScore += Mathf.Abs(3f - distanceFromTarget);
        }
    }

    // Take inputs for the BlackBox to feed into the NEAT algorithm.
    protected override void UpdateBlackBoxInputs(ISignalArray inputSignalArray)
    {
        VectorSensor sensor = new VectorSensor();

        InputSensors(sensor);
        double[] observations = sensor.Observations.ToArray();
        inputSignalArray.CopyFrom(observations, 0, observations.Length);

        var dt = Time.deltaTime;
        timeSinceStart += dt;

        var bpDict = m_JdController.bodyPartsDict;

        worstStandingZ = 0f;
        if (bpDict[footL].groundContact || bpDict[footR].groundContact)
        {
            if (onGroundScore < 1000)
            {
                onGroundScore++;
            }

            if (stepByStepScore < 100)
            {
                float footLZ = bpDict[footL].rb.position.z;
                float footRZ = bpDict[footR].rb.position.z;
                
                // Detect which feet are down and if the foot is different than the foot that got the previous best AND has a better Z score add a point and update the record.
                if (lastBestGroundedFoot != footL && bpDict[footL].groundContact)
                {
                    if (lastBestGroundedFootZ < footLZ)
                    {
                        updateStepByStep(footLZ);
                        //onGroundScore++;
                        lastBestGroundedFootZ = footLZ;
                        lastBestGroundedFoot = footL;
                    }
                }
                else if (lastBestGroundedFoot != footR && bpDict[footR].groundContact)
                {
                    if (lastBestGroundedFootZ < footRZ)
                    {
                        //onGroundScore++;
                        updateStepByStep(footRZ);
                        lastBestGroundedFootZ = footRZ;
                        lastBestGroundedFoot = footR;
                    }
                }
            }
        }

        if (bpDict[hips].rb.position.y < penaltyUnderY)
        {
            fallingOverPentalty--;
        }

        if (Mathf.Abs(CoM.x) > 3)
        {
            veeringPenalty--;
        }

        if (lastCoMZ < CoM.z)
        {
            timeSpentMoving += dt;
        }
        lastCoMZ = CoM.z;
    }

    // Use the output from the NEAT neural net to move the body.
    protected override void UseBlackBoxOutpts(ISignalArray outputSignalArray)
    {
        MoveBody(outputSignalArray);
    }

    // Calculate the fitness  based on totalDistanceTravelled, how much they remain on the ground, and timeSinceStart.
    public override float GetFitness()
    {
        var bpDict = m_JdController.bodyPartsDict;
        totalDistanceTraveled = bpDict[hips].rb.position.z;
        foreach (var part in m_JdController.bodyPartsList)
        {
            totalDistanceTraveled = Mathf.Min(totalDistanceTraveled, part.rb.position.z);
        }

        return Math.Max(0, 1
            + Math.Max(0, totalDistanceTraveled * distanceMultiplier)
            + Math.Max(0, (onGroundScore / 1000) * onGroundMultiplier)
            + Math.Max(0, timeSpentMoving * stayAliveMultiplier)
            + (fallingOverPentalty * fallingOverPenaltyMultiplier)
            + (veeringPenalty * veeringPenaltyMultiplier)
            + (stepByStepScore * stepByStepMultiplier));
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
    private void Awake()
    {
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
    public void Reset()
    {
        timeSinceStart = 0f;
        totalDistanceTraveled = 0f;
        avgVel = Vector3.zero;
        onGroundScore = 0f;
        fallingOverPentalty = 0f;
        lastBestGroundedFootZ = 0f;
        lastBestGroundedFoot = null;
        stepByStepScore = 0f;
        veeringPenalty = 0f;
        timeSpentMoving = 0f;

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
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
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {

        sensor.AddObservation(bp.groundContact.touchingGround);

        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));

        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));

        sensor.AddObservation(bp.rb.position);

        if (bp.rb.transform == hips) { return; }


        sensor.AddObservation(bp.rb.transform.localRotation);
        sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
    }

    // Gets all the input for the Neural Net.
    private void InputSensors(VectorSensor sensor)
    {
        avgVel = GetAvgVelocity();
        var bpDict = m_JdController.bodyPartsDict;

        sensor.AddObservation(avgVel);
        sensor.AddObservation(Mathf.Abs(bpDict[footL].rb.position.z - bpDict[footR].rb.position.z));

        // center of mass
        CoM = Vector3.zero;
        float c = 0f;

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
            CoM += bodyPart.rb.worldCenterOfMass * bodyPart.rb.mass;
            c += bodyPart.rb.mass;
        }

        CoM /= c;
        sensor.AddObservation(CoM);
    }

    // Sets the joints target and strength for all the joints. 
    public void MoveBody(ISignalArray continuousActions)
    {
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
