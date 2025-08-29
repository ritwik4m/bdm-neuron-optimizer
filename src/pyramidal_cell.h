// -----------------------------------------------------------------------------
//
// Simplified pyramidal cell simulation with exposed parameters
// One neuron grows, exported to SWC for analysis
//
// -----------------------------------------------------------------------------
#ifndef PYRAMIDAL_CELL_H_
#define PYRAMIDAL_CELL_H_

#include <fstream>
#include <iostream>
#include <string>
#include "biodynamo.h"
#include <ctime> 
#include "neuroscience/neuroscience.h"

namespace bdm {

// ----------------- Parameters + loader -----------------
struct GrowthParams {
  double elong_apical  = 80;    // µm per step (apical elongation distance)
  double elong_basal   = 50;    // µm per step (basal elongation distance)
  double branch_apical = 0.04;  // probability per step
  double branch_basal  = 0.006; // probability per step
  int    steps         = 500;   // number of scheduler steps
  double length_scale  = 1.0;   // scaling factor for elongation
};
static GrowthParams gParams;   // global instance accessible everywhere

inline void LoadParamsFromFile(const std::string& path = "bdm_params.txt") {
  std::ifstream in(path);
  if (!in) return; // use defaults if no file
  std::string key; double val;
  while (in >> key >> val) {
    if (key == "elong_apical")  gParams.elong_apical  = val;
    if (key == "elong_basal")   gParams.elong_basal   = val;
    if (key == "branch_apical") gParams.branch_apical = val;
    if (key == "branch_basal")  gParams.branch_basal  = val;
    if (key == "steps")         gParams.steps         = static_cast<int>(val);
    if (key == "length_scale")  gParams.length_scale  = val;
  }
}

// ----------------- Behaviors -----------------
struct ApicalDendriteGrowth : public Behavior {
  BDM_BEHAVIOR_HEADER(ApicalDendriteGrowth, Behavior, 1);
  ApicalDendriteGrowth() { AlwaysCopyToNew(); }
  virtual ~ApicalDendriteGrowth() {}

  void Run(Agent* agent) override {
    auto* sim = Simulation::GetActive();
    auto* random = sim->GetRandom();

    auto* dendrite = bdm_static_cast<NeuriteElement*>(agent);
    if (dendrite->GetDiameter() > 0.575) {
      auto old_direction = dendrite->GetSpringAxis() * 4;
      auto random_axis = random->template UniformArray<3>(-1, 1);
      auto new_step_direction = old_direction + random_axis * 0.3;

      dendrite->ElongateTerminalEnd(
        gParams.elong_apical * gParams.length_scale,
        new_step_direction);
      dendrite->SetDiameter(dendrite->GetDiameter() - 0.0007);

      if (dendrite->IsTerminal() && dendrite->GetDiameter() > 0.55 &&
          random->Uniform() < gParams.branch_apical) {
        auto rand_noise = random->template UniformArray<3>(-0.1, 0.1);
        auto branch_direction = dendrite->GetSpringAxis() + rand_noise;
        auto* dendrite_2 = dendrite->Branch(branch_direction);
        dendrite_2->SetDiameter(0.65);
      }
    }
  }
};

struct BasalDendriteGrowth : public Behavior {
  BDM_BEHAVIOR_HEADER(BasalDendriteGrowth, Behavior, 1);
  BasalDendriteGrowth() { AlwaysCopyToNew(); }
  virtual ~BasalDendriteGrowth() {}

  void Run(Agent* agent) override {
    auto* sim = Simulation::GetActive();
    auto* random = sim->GetRandom();

    auto* dendrite = bdm_static_cast<NeuriteElement*>(agent);
    if (dendrite->IsTerminal() && dendrite->GetDiameter() > 0.75) {
      auto old_direction = dendrite->GetSpringAxis() * 6;
      auto random_axis = random->template UniformArray<3>(-1, 1);
      auto new_step_direction = old_direction + random_axis * 0.4;

      dendrite->ElongateTerminalEnd(
        gParams.elong_basal * gParams.length_scale,
        new_step_direction);
      dendrite->SetDiameter(dendrite->GetDiameter() - 0.0008);

      if (random->Uniform() < gParams.branch_basal) {
        dendrite->Bifurcate();
      }
    }
  }
};

// ----------------- Create neuron -----------------
inline void AddInitialNeuron(const Real3& position) {
  auto* soma = new neuroscience::NeuronSoma(position);
  soma->SetDiameter(10);
  Simulation::GetActive()->GetResourceManager()->AddAgent(soma);

  auto* apical = soma->ExtendNewNeurite({0, 0, 1});
  auto* basal1 = soma->ExtendNewNeurite({0, 0, -1});
  auto* basal2 = soma->ExtendNewNeurite({0, 0.6, -0.8});
  auto* basal3 = soma->ExtendNewNeurite({0.3, -0.6, -0.8});

  apical->AddBehavior(new ApicalDendriteGrowth());
  basal1->AddBehavior(new BasalDendriteGrowth());
  basal2->AddBehavior(new BasalDendriteGrowth());
  basal3->AddBehavior(new BasalDendriteGrowth());
}

// ----------------- Save morphology -----------------
inline void SaveNeuronMorphology(Simulation& sim) {
  auto* rm = sim.GetResourceManager();
  rm->ForEachAgent([&](Agent* agent) {
    auto* soma = dynamic_cast<neuroscience::NeuronSoma*>(agent);
    if (soma != nullptr) {
      const char* trial_env = std::getenv("TRIAL_ID");
      std::string trial_id = trial_env ? trial_env : "unknown";
      std::string filename = "output/neuron_" + trial_id + ".swc";

      std::ofstream myfile(filename);
      soma->PrintSWC(myfile);
    }
  });
}

// ----------------- Simulation entry -----------------
inline int Simulate(int argc, const char** argv) {
  neuroscience::InitModule();
  Simulation simulation(argc, argv);

  LoadParamsFromFile();

  AddInitialNeuron({0, 0, 0});
  simulation.GetScheduler()->Simulate(gParams.steps);

  SaveNeuronMorphology(simulation);
  std::cout << "Simulation completed successfully!" << std::endl;
  return 0;
}

}  // namespace bdm

#endif  // PYRAMIDAL_CELL_H_
