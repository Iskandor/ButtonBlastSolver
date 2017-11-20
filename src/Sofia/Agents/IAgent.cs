using FLAB;
using Sofia.Algorithm.Exploration;

namespace Sofia
{
    public interface IAgent
    {
        void Init(NeuralNetwork p_network = null);
        void Dispose();
        Vector GetEstimate(Vector p_state0);
        int ChooseAction(IExploration p_exp, Vector p_estimate);
        void Train(Vector p_state0, Vector p_state1, float p_reward, bool p_finished);
        void Save(string p_filename);
        void Load(string p_filename);
        void SetParam(string p_param, int p_value);
        int GetParam(string p_param);
        Optimizer GetOptimizer();
    }
}
