﻿using System;
using BEPUphysics.BroadPhaseEntries;
using BEPUphysics.Entities.Prefabs;
using BEPUutilities;

namespace BEPUphysicsDemos.Demos
{
    /// <summary>
    /// Boxes fall onto a large terrain.  Try driving around on it!
    /// </summary>
    public class TerrainDemo : StandardDemo
    {
        /// <summary>
        /// Constructs a new demo.
        /// </summary>
        /// <param name="game">Game owning this demo.</param>
        public TerrainDemo(DemosGame game)
            : base(game)
        {
            //x and y, in terms of heightmaps, refer to their local x and y coordinates.  In world space, they correspond to x and z.
            //Setup the heights of the terrain.
            //[The size here is limited by the Reach profile the demos use- the drawer draws the terrain as a big block and runs into primitive drawing limits.
            //The physics can support far larger terrains!]
            int xLength = 180;
            int zLength = 180;

            float xSpacing = 8f;
            float zSpacing = 8f;
            var heights = new float[xLength, zLength];
            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < zLength; j++)
                {
                    float x = i - xLength / 2;
                    float z = j - zLength / 2;
                    //heights[i,j] = (float)(x * y / 1000f);
                    heights[i, j] = (float)(10 * (Math.Sin(x / 8) + Math.Sin(z / 8)));
                    //heights[i,j] = 3 * (float)Math.Sin(x * y / 100f);
                    //heights[i,j] = (x * x * x * y - y * y * y * x) / 1000f;
                }
            }
            //Create the terrain.
            var terrain = new Terrain(heights, new AffineTransform(
                    new Vector3(-xSpacing, 1, -zSpacing),
                    Quaternion.Identity,
                    new Vector3(xLength * xSpacing / 2, 0, zLength * zSpacing / 2)));
            terrain.Shape.QuadTriangleOrganization = BEPUphysics.CollisionShapes.QuadTriangleOrganization.BottomRightUpperLeft;

            //terrain.Thickness = 5; //Uncomment this and shoot some things at the bottom of the terrain! They'll be sucked up through the ground.


            Space.Add(terrain);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 5; k++)
                    {
                        Space.Add(new Box(
                            new Vector3(0 + i * 4, 100 - j * 10, 0 + k * 4),
                            2 + i * j * k,
                            2 + i * j * k,
                            2 + i * j * k,
                            4 + 20 * i * j * k));
                    }
                }
            }



            game.ModelDrawer.Add(terrain);

            game.Camera.Position = new Vector3(0, 30, 20);

        }





        /// <summary>
        /// Gets the name of the simulation.
        /// </summary>
        public override string Name
        {
            get { return "Terrain"; }
        }
    }
}