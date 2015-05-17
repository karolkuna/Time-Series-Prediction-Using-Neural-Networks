using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using BEPUphysics.BroadPhaseEntries.MobileCollidables;
using BEPUphysics.Constraints.SolverGroups;
using BEPUphysics.Constraints.TwoEntity.Motors;
using BEPUphysics.Entities;
using BEPUphysics.Entities.Prefabs;
using Microsoft.Xna.Framework.Input;
using BEPUphysics.CollisionRuleManagement;
using BEPUphysics.Materials;
using BEPUphysics.CollisionShapes;
using BEPUphysics.CollisionShapes.ConvexShapes;
using System.Collections.Generic;
using BEPUutilities;
using Microsoft.Xna.Framework.Graphics;
using ConversionHelper;
using System.Runtime.InteropServices;
using ManagedNeuralPredictionLibrary;

namespace BEPUphysicsDemos.Demos
{
    /// <summary>
    /// A clawbot spins, swivels, bends, and grabs.
    /// </summary>
    public class RobotArmDemo :
#if !WINDOWS
    //There's not enough buttons on the Xbox controller to have vehicles/characters and this demo's controls.
        Demo
#else
 StandardDemo
#endif
    {
        public RevoluteJoint groundToBaseJoint;
        public RevoluteJoint shoulder;
        public SwivelHingeJoint elbow;
        public RevoluteJoint clawHingeA;
        public RevoluteJoint clawHingeB;
        public Entity lowerArm;
        public Entity upperArm;
        public CompoundBody leftClaw;
        public CompoundBody rightClaw;

        public ManagedMemoryBlock input;
        public ManagedMemoryBlock target;
        public ManagedLogisticFunction logisticFunction;
        public ManagedSimpleRecurrentNetwork network;
        public ManagedTBPTT learningAlgorithm;

        public float predictedX, predictedY, predictedZ;

        /// <summary>
        /// Gets the name of the simulation.
        /// </summary>
        public override string Name
        {
            get { return "Robotic Arm Thingamajig"; }
        }

        /// <summary>
        /// Constructs a new demo.
        /// </summary>
        /// <param name="game">Game owning this demo.</param>
        public RobotArmDemo(DemosGame game)
            : base(game)
        {
            //Since this is not a "StandardDemo" derived class, we need to set our own gravity.
            Space.ForceUpdater.Gravity = new Vector3(0, -9.81f, 0);
            Entity ground = new Box(Vector3.Zero, 30, 1, 30);
            Space.Add(ground);

            var armBase = new Cylinder(new Vector3(0, 2, 0), 2.5f, 1, 40);
            Space.Add(armBase);

            //The arm base can rotate around the Y axis.
            //Rotation is controlled by user input.
            groundToBaseJoint = new RevoluteJoint(ground, armBase, Vector3.Zero, Vector3.Up);
            groundToBaseJoint.Motor.IsActive = true;
            groundToBaseJoint.Motor.Settings.Mode = MotorMode.VelocityMotor;
            groundToBaseJoint.Motor.Settings.MaximumForce = 3500;
            Space.Add(groundToBaseJoint);

            lowerArm = new Box(armBase.Position + new Vector3(0, 2, 0), 1, 3, .5f, 10);
            Space.Add(lowerArm);

            shoulder = new RevoluteJoint(armBase, lowerArm, armBase.Position, Vector3.Forward);
            shoulder.Motor.IsActive = true;
            shoulder.Motor.Settings.Mode = MotorMode.VelocityMotor;
            shoulder.Motor.Settings.MaximumForce = 2500;

            //Don't want it to rotate too far; this keeps the whole contraption off the ground.
            shoulder.Limit.IsActive = true;
            shoulder.Limit.MinimumAngle = 0;
            shoulder.Limit.MaximumAngle = MathHelper.PiOver4;
            Space.Add(shoulder);

            //Make the next piece of the arm.
            upperArm = new Cylinder(lowerArm.Position + new Vector3(0, 3, 0), 3, .25f, 10);
            Space.Add(upperArm);

            //Swivel hinges allow motion around two axes.  Imagine a tablet PC's monitor hinge.
            elbow = new SwivelHingeJoint(lowerArm, upperArm, lowerArm.Position + new Vector3(0, 1.5f, 0), Vector3.Forward);
            elbow.TwistMotor.IsActive = true;
            elbow.TwistMotor.Settings.Mode = MotorMode.VelocityMotor;
            elbow.TwistMotor.Settings.MaximumForce = 1000;

            elbow.HingeMotor.IsActive = true;
            elbow.HingeMotor.Settings.Mode = MotorMode.VelocityMotor;
            elbow.HingeMotor.Settings.MaximumForce = 1250;

            //Keep it from rotating too much.
            elbow.HingeLimit.IsActive = true;
            elbow.HingeLimit.MinimumAngle = 0;
            elbow.HingeLimit.MaximumAngle = MathHelper.PiOver2;
            Space.Add(elbow);


            //Add a menacing claw at the end.
            var lowerPosition = upperArm.Position + new Vector3(-.65f, 1.6f, 0);

            CollisionRules clawPart1ARules = new CollisionRules();
            var bodies = new List<CompoundChildData>()
            {
                new CompoundChildData() { Entry = new CompoundShapeEntry(new BoxShape(1, .25f, .25f), lowerPosition, 3), CollisionRules = clawPart1ARules },
                new CompoundChildData() { Entry = new CompoundShapeEntry(new ConeShape(1, .125f), lowerPosition + new Vector3(-.375f, .4f, 0), 3), Material = new Material(2,2,0) }
            };

            leftClaw = new CompoundBody(bodies, 6);
            Space.Add(leftClaw);

            clawHingeA = new RevoluteJoint(upperArm, leftClaw, upperArm.Position + new Vector3(0, 1.5f, 0), Vector3.Forward);
            clawHingeA.Motor.IsActive = true;
            clawHingeA.Motor.Settings.Mode = MotorMode.VelocityMotor;
            clawHingeA.Motor.Settings.Servo.Goal = -MathHelper.PiOver2;
            //Weaken the claw to prevent it from crushing the boxes.
            clawHingeA.Motor.Settings.Servo.SpringSettings.Damping /= 100;
            clawHingeA.Motor.Settings.Servo.SpringSettings.Stiffness /= 100;

            clawHingeA.Limit.IsActive = true;
            clawHingeA.Limit.MinimumAngle = -MathHelper.PiOver2;
            clawHingeA.Limit.MaximumAngle = -MathHelper.Pi / 6;
            Space.Add(clawHingeA);

            //Add one more claw to complete the contraption.
            lowerPosition = upperArm.Position + new Vector3(.65f, 1.6f, 0);

            CollisionRules clawPart1BRules = new CollisionRules();
            bodies = new List<CompoundChildData>()
            {
                new CompoundChildData() { Entry = new CompoundShapeEntry(new BoxShape(1, .25f, .25f), lowerPosition, 3), CollisionRules = clawPart1BRules },
                new CompoundChildData() { Entry = new CompoundShapeEntry(new ConeShape(1, .125f), lowerPosition + new Vector3(.375f, .4f, 0), 3), Material = new Material(2,2,0) }
            };

            rightClaw = new CompoundBody(bodies, 6);
            Space.Add(rightClaw);

            clawHingeB = new RevoluteJoint(upperArm, rightClaw, upperArm.Position + new Vector3(0, 1.5f, 0), Vector3.Forward);
            clawHingeB.Motor.IsActive = true;
            clawHingeB.Motor.Settings.Mode = MotorMode.VelocityMotor;
            clawHingeB.Motor.Settings.Servo.Goal = MathHelper.PiOver2;
            //Weaken the claw to prevent it from crushing the boxes.
            clawHingeB.Motor.Settings.Servo.SpringSettings.Damping /= 100;
            clawHingeB.Motor.Settings.Servo.SpringSettings.Stiffness /= 100;

            clawHingeB.Limit.IsActive = true;
            clawHingeB.Limit.MinimumAngle = MathHelper.Pi / 6;
            clawHingeB.Limit.MaximumAngle = MathHelper.PiOver2;
            Space.Add(clawHingeB);

            //Keep the pieces of the robot from interacting with each other.
            //The CollisionRules.AddRule method is just a convenience method that adds items to the 'specific' dictionary.
            //Sometimes, it's a little unwieldy to reference the collision rules,
            //so the convenience method just takes the owners and hides the ugly syntax.
            CollisionRules.AddRule(armBase, lowerArm, CollisionRule.NoBroadPhase);
            CollisionRules.AddRule(lowerArm, upperArm, CollisionRule.NoBroadPhase);
            CollisionRules.AddRule(clawPart1ARules, upperArm, CollisionRule.NoBroadPhase);
            CollisionRules.AddRule(clawPart1BRules, upperArm, CollisionRule.NoBroadPhase);
            //Here's an example without a convenience method.  Since they are both direct CollisionRules references, it's pretty clean.
            clawPart1BRules.Specific.Add(clawPart1ARules, CollisionRule.NoBroadPhase);


            //Put some boxes on the ground to try to pick up.
            for (double k = 0; k < Math.PI * 2; k += Math.PI / 6)
            {
                //Space.Add(new Box(new Vector3((float)Math.Cos(k) * 5.5f, 2, (float)Math.Sin(k) * 5.5f), 1, 1, 1, 10));
            }

            game.Camera.Position = new Vector3(0, 5, 13);



            logisticFunction = new ManagedLogisticFunction();
            input = new ManagedMemoryBlock(9);
            target = new ManagedMemoryBlock(3);

            network = new ManagedSimpleRecurrentNetwork(9, 256, 3, logisticFunction);
            learningAlgorithm = new ManagedTBPTT(network, 0.005f, 0.9f, 4);
        }


        Random rnd = new Random();
        int currentStep = 0;
        NumberFormatInfo nfi = new CultureInfo("en-US", false).NumberFormat;

        public float GetJointAngle(SwivelHingeJoint joint)
        {
            float angle = (float)Math.Acos(Vector3.Dot(Vector3.Normalize(joint.BallSocketJoint.OffsetA), Vector3.Normalize(joint.BallSocketJoint.OffsetB)));
            if (float.IsNaN(angle) || float.IsInfinity(angle))
                angle = 0;
            return angle;
        }

        public float GetJointAngle(RevoluteJoint joint)
        {
            float angle = (float)Math.Acos(Vector3.Dot(Vector3.Normalize(joint.AngularJoint.ConnectionA.OrientationMatrix.Up), Vector3.Normalize(joint.AngularJoint.ConnectionB.OrientationMatrix.Up)));
            if (float.IsNaN(angle) || float.IsInfinity(angle))
                angle = 0;
            return angle;
        }


        public override void Update(float dt)
        {
#if !WINDOWS
            if (Game.GamePadInput.IsButtonDown(Buttons.LeftShoulder))
                groundToBaseJoint.Motor.Settings.Servo.Goal -= 1 * dt;
            if (Game.GamePadInput.IsButtonDown(Buttons.RightShoulder))
                groundToBaseJoint.Motor.Settings.Servo.Goal += 1 * dt;

            if (Game.GamePadInput.IsButtonDown(Buttons.A))
                shoulder.Motor.Settings.Servo.Goal = MathHelper.Min(shoulder.Motor.Settings.Servo.Goal + .5f * dt, shoulder.Limit.MaximumAngle);
            if (Game.GamePadInput.IsButtonDown(Buttons.B))
                shoulder.Motor.Settings.Servo.Goal = MathHelper.Max(shoulder.Motor.Settings.Servo.Goal - .5f * dt, shoulder.Limit.MinimumAngle);

            if (Game.GamePadInput.IsButtonDown(Buttons.X))
                elbow.HingeMotor.Settings.Servo.Goal = MathHelper.Min(elbow.HingeMotor.Settings.Servo.Goal + 1 * dt, elbow.HingeLimit.MaximumAngle);
            if (Game.GamePadInput.IsButtonDown(Buttons.Y))
                elbow.HingeMotor.Settings.Servo.Goal = MathHelper.Max(elbow.HingeMotor.Settings.Servo.Goal - 1 * dt, elbow.HingeLimit.MinimumAngle);

            if (Game.GamePadInput.IsButtonDown(Buttons.DPadUp))
                elbow.TwistMotor.Settings.Servo.Goal += 1f * dt;
            if (Game.GamePadInput.IsButtonDown(Buttons.DPadDown))
                elbow.TwistMotor.Settings.Servo.Goal -= 1f * dt;

            if (Game.GamePadInput.IsButtonDown(Buttons.LeftTrigger))
            {
                clawHingeA.Motor.Settings.Servo.Goal = MathHelper.Max(clawHingeA.Motor.Settings.Servo.Goal - Game.GamePadInput.Triggers.Left * 1.5f * dt, clawHingeA.Limit.MinimumAngle);
                clawHingeB.Motor.Settings.Servo.Goal = MathHelper.Min(clawHingeB.Motor.Settings.Servo.Goal + Game.GamePadInput.Triggers.Left * 1.5f * dt, clawHingeB.Limit.MaximumAngle);
            }
            if (Game.GamePadInput.IsButtonDown(Buttons.RightTrigger))
            {
                clawHingeA.Motor.Settings.Servo.Goal = MathHelper.Min(clawHingeA.Motor.Settings.Servo.Goal + Game.GamePadInput.Triggers.Right * 1.5f * dt, clawHingeA.Limit.MaximumAngle);
                clawHingeB.Motor.Settings.Servo.Goal = MathHelper.Max(clawHingeB.Motor.Settings.Servo.Goal - Game.GamePadInput.Triggers.Right * 1.5f * dt, clawHingeB.Limit.MinimumAngle);
            }

#else

            currentStep++;

            if (currentStep % 60 == 0)
            {
                groundToBaseJoint.Motor.Settings.VelocityMotor.GoalVelocity = 2 * ((float)(rnd.NextDouble()) - 0.5f);
                shoulder.Motor.Settings.VelocityMotor.GoalVelocity = 2 * ((float)(rnd.NextDouble()) - 0.5f);
                elbow.HingeMotor.Settings.VelocityMotor.GoalVelocity = 2 * ((float)(rnd.NextDouble()) - 0.5f);
                //clawHingeA.Motor.Settings.VelocityMotor.GoalVelocity = 2 * ((float)(rnd.NextDouble()) - 0.5f);
                //clawHingeB.Motor.Settings.VelocityMotor.GoalVelocity = clawHingeA.Motor.Settings.Servo.Goal;
            }
            else
            {
                groundToBaseJoint.Motor.Settings.VelocityMotor.GoalVelocity += 0.1f * ((float)(rnd.NextDouble()) - 0.5f);
                shoulder.Motor.Settings.VelocityMotor.GoalVelocity += 0.1f * ((float)(rnd.NextDouble()) - 0.5f);
                elbow.HingeMotor.Settings.VelocityMotor.GoalVelocity += 0.1f * ((float)(rnd.NextDouble()) - 0.5f);
                //clawHingeA.Motor.Settings.VelocityMotor.GoalVelocity += 0.1f * ((float)(rnd.NextDouble()) - 0.5f);
                //clawHingeB.Motor.Settings.VelocityMotor.GoalVelocity = clawHingeA.Motor.Settings.Servo.Goal;
            }


            float baseRotation = 0.5f * groundToBaseJoint.Motor.ConnectionB.Orientation.Y + 0.5f;
            float shoulderRotation = GetJointAngle(shoulder);
            float elbowRotation = GetJointAngle(elbow) / MathHelper.Pi - 0.25f;
            var center = (leftClaw.Position + rightClaw.Position) * 0.5f;
            float x = 0.5f + center.X * 0.075f;
            float y = center.Y * 0.1f;
            float z = 0.5f + center.Z * 0.075f;

            target.SetValue(0, x);
            target.SetValue(1, y);
            target.SetValue(2, z);

            input.SetValue(0, x);
            input.SetValue(1, y);
            input.SetValue(2, z);
            input.SetValue(3, baseRotation);
            input.SetValue(4, shoulderRotation);
            input.SetValue(5, elbowRotation);
            input.SetValue(6, 0.5f + 0.5f*groundToBaseJoint.Motor.Settings.VelocityMotor.GoalVelocity);
            input.SetValue(7, 0.5f + 0.5f*shoulder.Motor.Settings.VelocityMotor.GoalVelocity);
            input.SetValue(8, 0.5f + 0.5f * elbow.HingeMotor.Settings.VelocityMotor.GoalVelocity);

            learningAlgorithm.Train(target);
            network.Propagate(input);

            predictedX = (network.GetOutputValue(0) - 0.5f) * (1.0f/0.075f);
            predictedY = network.GetOutputValue(1) * 10f;
            predictedZ = (network.GetOutputValue(2) - 0.5f) * (1.0f / 0.075f);

            /*using (StreamWriter w = File.AppendText(@"C:\Users\Karol\Desktop 2\manipulator.txt"))
            {
                w.WriteLine(groundToBaseJoint.Motor.Settings.VelocityMotor.GoalVelocity.ToString("F6", nfi) + " " + shoulder.Motor.Settings.VelocityMotor.GoalVelocity.ToString("F6", nfi) + " " + elbow.HingeMotor.Settings.VelocityMotor.GoalVelocity.ToString("F6", nfi));
                w.WriteLine(baseRotation.ToString("F6", nfi) + " " + shoulderRotation.ToString("F6", nfi) + " " + elbowRotation.ToString("F6", nfi) + " " + x.ToString("F6", nfi) + " " + y.ToString("F6", nfi) + " " + z.ToString("F6", nfi));
            }*/


            /*
            if (Game.KeyboardInput.IsKeyDown(Keys.N))
                groundToBaseJoint.Motor.Settings.Servo.Goal -= 1 * dt;
            if (Game.KeyboardInput.IsKeyDown(Keys.M))
                groundToBaseJoint.Motor.Settings.Servo.Goal += 1 * dt;

            if (Game.KeyboardInput.IsKeyDown(Keys.Q))
                shoulder.Motor.Settings.Servo.Goal = MathHelper.Min(shoulder.Motor.Settings.Servo.Goal + .5f * dt, shoulder.Limit.MaximumAngle);
            if (Game.KeyboardInput.IsKeyDown(Keys.W))
                shoulder.Motor.Settings.Servo.Goal = MathHelper.Max(shoulder.Motor.Settings.Servo.Goal - .5f * dt, shoulder.Limit.MinimumAngle);

            if (Game.KeyboardInput.IsKeyDown(Keys.R))
                elbow.HingeMotor.Settings.Servo.Goal = MathHelper.Min(elbow.HingeMotor.Settings.Servo.Goal + 1 * dt, elbow.HingeLimit.MaximumAngle);
            if (Game.KeyboardInput.IsKeyDown(Keys.T))
                elbow.HingeMotor.Settings.Servo.Goal = MathHelper.Max(elbow.HingeMotor.Settings.Servo.Goal - 1 * dt, elbow.HingeLimit.MinimumAngle);

            if (Game.KeyboardInput.IsKeyDown(Keys.O))
                elbow.TwistMotor.Settings.Servo.Goal += 1f * dt;
            if (Game.KeyboardInput.IsKeyDown(Keys.P))
                elbow.TwistMotor.Settings.Servo.Goal -= 1f * dt;

            if (Game.KeyboardInput.IsKeyDown(Keys.OemOpenBrackets))
            {
                clawHingeA.Motor.Settings.Servo.Goal = MathHelper.Max(clawHingeA.Motor.Settings.Servo.Goal - 1.5f * dt, clawHingeA.Limit.MinimumAngle);
                clawHingeB.Motor.Settings.Servo.Goal = MathHelper.Min(clawHingeB.Motor.Settings.Servo.Goal + 1.5f * dt, clawHingeB.Limit.MaximumAngle);
            }
            if (Game.KeyboardInput.IsKeyDown(Keys.OemCloseBrackets))
            {
                clawHingeA.Motor.Settings.Servo.Goal = MathHelper.Min(clawHingeA.Motor.Settings.Servo.Goal + 1.5f * dt, clawHingeA.Limit.MaximumAngle);
                clawHingeB.Motor.Settings.Servo.Goal = MathHelper.Max(clawHingeB.Motor.Settings.Servo.Goal - 1.5f * dt, clawHingeB.Limit.MinimumAngle);
            }
            */

#endif
            base.Update(dt);
        }

        public override void DrawUI()
        {
            base.DrawUI();
            Game.DataTextDrawer.Draw("Arm controls:", new Microsoft.Xna.Framework.Vector2(50, 20));
#if !WINDOWS
            Game.TinyTextDrawer.Draw("Spin base: Left/Right Shoulder", new Vector2(50, 38));
            Game.TinyTextDrawer.Draw("Bend shoulder: A B", new Vector2(50, 53));
            Game.TinyTextDrawer.Draw("Bend elbow: X Y", new Vector2(50, 68));
            Game.TinyTextDrawer.Draw("Spin forearm: Up/Down Dpad", new Vector2(50, 83));
            Game.TinyTextDrawer.Draw("Open/close claw: Left/Right Trigger", new Vector2(50, 98));
#else
            Game.TinyTextDrawer.Draw("Spin base: N M", new Microsoft.Xna.Framework.Vector2(50, 38));
            Game.TinyTextDrawer.Draw("Bend shoulder: Q W", new Microsoft.Xna.Framework.Vector2(50, 53));
            Game.TinyTextDrawer.Draw("Bend elbow: R T", new Microsoft.Xna.Framework.Vector2(50, 68));
            Game.TinyTextDrawer.Draw("Spin forearm: O P", new Microsoft.Xna.Framework.Vector2(50, 83));
            Game.TinyTextDrawer.Draw("Open/close claw: [ ]", new Microsoft.Xna.Framework.Vector2(50, 98));

            float baseRotation = 0.5f * groundToBaseJoint.Motor.ConnectionB.Orientation.Y + 0.5f;
            float shoulderRotation = GetJointAngle(shoulder);
            float elbowRotation = GetJointAngle(elbow) / MathHelper.Pi - 0.25f;
            var center = (leftClaw.Position + leftClaw.Position) * 0.5f;
            float x = 0.5f + center.X * 0.075f;
            float y = center.Y * 0.1f;
            float z = 0.5f + center.Z * 0.075f;

            Game.TinyTextDrawer.Draw(baseRotation.ToString("F6", nfi) + " " + shoulderRotation.ToString("F6", nfi) + " " + elbowRotation.ToString("F6", nfi) + " " + x.ToString("F6", nfi) + " " + y.ToString("F6", nfi) + " " + z.ToString("F6", nfi), new Microsoft.Xna.Framework.Vector2(50, 120));
#endif
        }

        VertexPositionColor[] lines = new VertexPositionColor[] { new VertexPositionColor(new Microsoft.Xna.Framework.Vector3(), Microsoft.Xna.Framework.Color.Red), new VertexPositionColor(new Microsoft.Xna.Framework.Vector3(), Microsoft.Xna.Framework.Color.White) };
        public override void Draw()
        {
            base.Draw();

            lines[0].Position = MathConverter.Convert(0.5f * (leftClaw.Position + rightClaw.Position));
            lines[1].Position = MathConverter.Convert(new Vector3(predictedX, predictedY, predictedZ));
            foreach (EffectPass pass in Game.LineDrawer.CurrentTechnique.Passes)
            {
                pass.Apply();

                Game.GraphicsDevice.DrawUserPrimitives(PrimitiveType.LineList, lines, 0, lines.Length / 2);
            }
        }
    }
}
