using FLAB;
using System.Collections;

namespace Sofia
{
    public class A3C : ActorCritic
    {
        private struct StackItem
        {
            public Vector State { get; set; }
            public int Action { get; set; }
            public float Reward { get; set; }
        }

        private NeuralNetwork   _thNetwork;
        private Optimizer       _thOptimizer;
        private AsyncUnit       _asyncUnit;

        private Stack _stack;

        public A3C(Optimizer p_optimizer, NeuralNetwork p_network, float p_gamma, int p_asynchUpdate) : base(p_optimizer, p_network, p_gamma)
        {
            _thNetwork = IOUtils.LoadNetwork(IOUtils.SaveNetwork(p_network));
            _thOptimizer = new ADAM(_thNetwork);

            _asyncUnit = new AsyncUnit();
            _asyncUnit.Init(p_asynchUpdate);
            _thOptimizer.InitAsynchMode(true);

            _stack = new Stack();
        }

        override public double Train(Vector p_state0, int p_action, Vector p_state1, float p_reward, bool p_final)
        {
            double mse = 0;

            _stack.Push(new StackItem { State = Vector.Copy(p_state0), Action = p_action, Reward = p_reward });

            _asyncUnit.Update(p_final);

            if (_asyncUnit.IsAsyncReady)
            {
                int criticOutput = _thNetwork.Output.Size - 1;
                float value1 = _thNetwork.Activate(p_state1)[criticOutput];

                float R = p_final ? 0f : value1;

                while(_stack.Count > 0)
                {
                    StackItem item = (StackItem)_stack.Pop();
                    R = item.Reward + _gamma * R;

                    Vector target = Vector.Copy(_thNetwork.Activate(item.State));
                    
                    float value0 = _thNetwork.Activate(p_state0)[criticOutput];

                    target[p_action] = R - value0;
                    target[criticOutput] = R;

                    mse += _thOptimizer.Train(item.State, target);

                    Vector.Release(item.State);
                    Vector.Release(target);
                }

                _optimizer.AsyncUpdate(_thOptimizer);
                _thNetwork.OverrideParams(_network);
            }

            return mse;
        }

        public override void SetAlpha(float p_alpha)
        {
            _thOptimizer.Alpha = p_alpha;
        }
    }
}
