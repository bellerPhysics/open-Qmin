#ifndef SIMPLEMODEL_H
#define SIMPLEMODEL_H

#include "std_include.h"
#include "gpuarray.h"
#include "periodicBoundaryConditions.h"
#include "functions.h"
#include "noiseSource.h"

/*! \file simpleModel.h
 * \brief defines an interface for models that compute forces
 */

//! A base interfacing class that defines common operations
/*!
This provides an interface, guaranteeing that SimpleModel S will provide access to
S.setGPU();
S.getNumberOfParticles();
S.computeForces();
S.moveParticles();
S.returnForces();
S.returnPositions();
S.returnVelocities();
S.returnMasses();
S.spatialSorting();
S.returnAdditionalData();
*/
class simpleModel
    {
    public:
        //!The base constructor requires the number of particles
        simpleModel(int n, bool _useGPU = false);
        //!initialize the size of the basic data structure arrays
        void initializeSimpleModel(int n);

        //!Enforce GPU operation
        virtual void setGPU(bool _useGPU){useGPU = _useGPU;};
        //!get the number of degrees of freedom, defaulting to the number of cells
        virtual int getNumberOfParticles(){return N;};
        //!move the degrees of freedom
        virtual void moveParticles(GPUArray<dVec> &displacements,scalar scale = 1.);
        //!do everything unusual to compute additional forces... by default, sets forces to zero
        virtual void computeForces(bool zeroOutForces=false);

        void setParticlePositionsRandomly(noiseSource &noise);

        //!do everything necessary to perform a Hilbert sort
        virtual void spatialSorting(){};

        //!return a reference to the GPUArray of positions
        virtual GPUArray<dVec> & returnPositions(){return positions;};
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<dVec> & returnForces(){return forces;};
        //!return a reference to the GPUArray of the masses
        virtual GPUArray<scalar> & returnMasses(){return masses;};
        //!return a reference to the GPUArray of the current velocities
        virtual GPUArray<dVec> & returnVelocities(){return velocities;};

        //!Does this model have a special force it needs to compute itself?
        bool selfForceCompute;

        //!The space in which the particles live
        BoxPtr Box;

    protected:
        //!The number of particles
        int N;
        //!particle  positions
        GPUArray<dVec> positions;
        //!particle velocities
        GPUArray<dVec> velocities;
        //!Forces on particles
        GPUArray<dVec> forces;
        //!particle masses
        GPUArray<scalar> masses;

        //!Whether the GPU should be used to compute anything
        bool useGPU;

    };
typedef shared_ptr<simpleModel> ConfigPtr;
typedef weak_ptr<simpleModel> WeakConfigPtr;
#endif
