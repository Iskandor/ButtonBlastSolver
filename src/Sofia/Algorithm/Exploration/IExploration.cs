using FLAB;

namespace Sofia.Algorithm.Exploration
{
    public interface IExploration
    {
        void Init(params float[] p_params);
        int  ChooseAction(Vector p_estimates, bool p_probabilities = false);
        void UpdateParams(float p_f);
        string ToString();
    }
}
