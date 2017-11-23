using FLAB;
using Sofia;
using Sofia.Algorithm.Exploration;
using Sofia.Layers;

public class ACN : BaseAgent
{
    private ActorCritic _actorCritic;
    private int _action;

    override public void Init(NeuralNetwork p_network = null)
    {
        NeuralNetwork network = null;
        Optimizer optimizer = null;

        if (p_network == null)
        {
            network = new NeuralNetwork();
            network.AddLayer("input", new InputLayer(GetParam(STATE_DIM)), BaseLayer.TYPE.INPUT);
            network.AddLayer("hidden0", new CoreLayer(SolverConfig.GetInstance().hidden_layer, ACTIVATION.RELU, BaseLayer.TYPE.HIDDEN), BaseLayer.TYPE.HIDDEN);
            network.AddLayer("output", new CoreLayer(GetParam(ACTION_DIM) + 1, ACTIVATION.TANH, BaseLayer.TYPE.OUTPUT), BaseLayer.TYPE.OUTPUT);

            // feed-forward connections
            network.AddConnection("input", "hidden0", Connection.INIT.GLOROT_UNIFORM);
            network.AddConnection("hidden0", "output", Connection.INIT.GLOROT_UNIFORM);
        }
        else
        {
            network = p_network;
        }

        optimizer = new ADAM(network);
        //optimizer.InitAsynchMode(true);
        //optimizer = new RMSProp(network);
        //optimizer = new BackProp(network, 1e-5f, 0.99f, true);
        //_actorCritic = new A3C(optimizer, network, 0.99f, SolverConfig.GetInstance().async_update);
        _actorCritic = new ActorCritic(optimizer, network, 0.99f);
        _actorCritic.SetAlpha(SolverConfig.GetInstance().learning_rate);
    }

    override public void Dispose()
    {
        _actorCritic.Dispose();
    }

    override public Vector GetEstimate(Vector p_state0)
    {
        Vector o = _actorCritic.Activate(p_state0);
        Vector e = Vector.Zero(o.Size - 1);

        for(int i = 0; i < e.Size; i++)
        {
            e[i] = o[i];
        }

        return e;
    }

    override public int ChooseAction(IExploration p_exp, Vector p_estimate)
    {
        _action = p_exp.ChooseAction(p_estimate, false);
        return _action;
    }

    override public void Train(Vector p_state0, Vector p_state1, float p_reward, bool p_finished)
    {
        _actorCritic.Train(p_state0, _action, p_state1, p_reward, p_finished);
    }

    public void ResetContext()
    {
        _actorCritic.Network.ResetContext();
    }

    override public Optimizer GetOptimizer()
    {
        return _actorCritic.Optimizer;
    }
}
