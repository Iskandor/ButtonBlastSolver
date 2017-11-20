using FLAB;
using Sofia;
using Sofia.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class XOR
{
    public XOR()
    {

    }

    public void Run()
    {
        Stopwatch watch = Stopwatch.StartNew();

        NeuralNetwork network = new NeuralNetwork();

        network.AddLayer("input", new InputLayer(2), BaseLayer.TYPE.INPUT);
        network.AddLayer("hidden", new CoreLayer(8, ACTIVATION.SIGMOID, BaseLayer.TYPE.HIDDEN), BaseLayer.TYPE.HIDDEN);
        network.AddLayer("output", new CoreLayer(1, ACTIVATION.SIGMOID, BaseLayer.TYPE.OUTPUT), BaseLayer.TYPE.OUTPUT);
        network.AddConnection("input", "hidden", Connection.INIT.GLOROT_UNIFORM);
        network.AddConnection("hidden", "output", Connection.INIT.GLOROT_UNIFORM);

        /*
        Optimizer optimizer = new BackProp(network, 1e-5f, 0.99f, true)
        {
            Alpha = 0.1f
        };
        */

        Optimizer optimizer = new RMSProp(network)
        {
            Alpha = 0.1f
        };

        optimizer.InitBatchMode(4);

        Vector[] input = new Vector[4];
        Vector[] target = new Vector[4];
        //Vector output = null;

        input[0] = Vector.Build(2, new float[] { 0f, 0f });
        input[1] = Vector.Build(2, new float[] { 0f, 1f });
        input[2] = Vector.Build(2, new float[] { 1f, 0f });
        input[3] = Vector.Build(2, new float[] { 1f, 1f });

        target[0] = Vector.Build(1, new float[] { 0f });
        target[1] = Vector.Build(1, new float[] { 1f });
        target[2] = Vector.Build(1, new float[] { 1f });
        target[3] = Vector.Build(1, new float[] { 0f });


        for (int e = 0; e < 200; e++)
        {
            //Console.Write("Start ");
            //BasePool.Instance.Check();

            float err = 0;

            for (int i = 0; i < 4; i++)
            {
                err += optimizer.Train(input[i], target[i]);
            }
            
            Console.WriteLine(err);

            //Console.Write("End ");
            //BasePool.Instance.Check();
        }
        Console.WriteLine();

        for (int i = 0; i < 4; i++)
        {
            Console.WriteLine(network.Activate(input[i])[0]);
            Vector.Release(input[i]);
            Vector.Release(target[i]);
        }

        optimizer.Dispose();

        Console.Write("Finish ");
        BasePool.Instance.Check();

        watch.Stop();
        Console.WriteLine(watch.ElapsedMilliseconds);
    }
}
