using Sofia;
using FLAB;
using Sofia.Layers;
using System;
using Sofia.Algorithm.Exploration;

public class DQN : BaseAgent
{
    private DeepQLearning _critic;

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
            network.AddLayer("output", new CoreLayer(GetParam(ACTION_DIM), ACTIVATION.TANH, BaseLayer.TYPE.OUTPUT), BaseLayer.TYPE.OUTPUT);

            // feed-forward connections
            network.AddConnection("input", "hidden0", Connection.INIT.GLOROT_UNIFORM);
            network.AddConnection("hidden0", "output", Connection.INIT.GLOROT_UNIFORM);
        }
        else
        {
            network = p_network;
        }

        optimizer = new ADAM(network);
        //optimizer = new RMSProp(network);
        //optimizer = new BackProp(network, 1e-5f, 0.99f, true);
        _critic = new DeepQLearning(optimizer, network, 0.99f, SolverConfig.GetInstance().memory_size, SolverConfig.GetInstance().batch_size, SolverConfig.GetInstance().qtupdate_size);
        _critic.SetAlpha(SolverConfig.GetInstance().learning_rate);
    }

    override public void Dispose()
    {
        _critic.Dispose();
    }

    override public Vector GetEstimate(Vector p_state0)
    {
        return _critic.Activate(p_state0);
    }

    override public int ChooseAction(IExploration p_exp, Vector p_estimate)
    {
        _action = p_exp.ChooseAction(p_estimate);
        return _action;
    }

    override public void Train(Vector p_state0, Vector p_state1, float p_reward, bool p_finished)
    {
        _critic.Train(p_state0, _action, p_state1, p_reward, p_finished);
    }

    public void ResetContext()
    {
        _critic.Network.ResetContext();
    }

    override public Optimizer GetOptimizer()
    {
        return _critic.Optimizer;
    }

    override public void Save(string p_filename)
    {
        string data = null;

        data = IOUtils.SaveNetwork(_critic.Network);
        System.IO.File.WriteAllText(@".\" + p_filename, data);
    }

    override public void Load(string p_filename)
    {
        NeuralNetwork network = new NeuralNetwork();
        string data = null;

        data = System.IO.File.ReadAllText(@".\" + p_filename);
        network = IOUtils.LoadNetwork(data);

        Init(network);
    }
}