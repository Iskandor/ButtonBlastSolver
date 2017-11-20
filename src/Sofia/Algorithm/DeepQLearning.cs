using FLAB;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Sofia
{
    public class DeepQLearning
    {
        public struct ReplayBufferElem
        {
            public Vector S0 { get; set; }
            public Vector S1 { get; set; }
            public int Action { get; set; }
            public float Reward { get; set; }
            public bool Final { get; set; }
        }

        protected IEnumerable<ReplayBufferElem> _sample;
        protected Queue<ReplayBufferElem> _replayBuffer;
        protected int _replayBufferCapacity;
        protected int _batchIndex, _batchSize;
        protected Random _rnd;
        protected int _qtUpdateIndex, _qtUpdateSize;

        protected NeuralNetwork _Qnetwork;
        protected NeuralNetwork _QTnetwork;
        protected Optimizer _optimizer;
        protected float _gamma;

        public DeepQLearning(Optimizer p_optimizer, NeuralNetwork p_network, float p_gamma, int p_capacity, int p_batchSize, int p_qtUpdateSize)
        {
            _optimizer = p_optimizer;
            _optimizer.InitBatchMode(p_batchSize);
            _Qnetwork = p_network;
            _gamma = p_gamma;

            _qtUpdateIndex = 0;
            _qtUpdateSize = p_qtUpdateSize;

            _rnd = new Random();

            _batchIndex = 0;
            _batchSize = p_batchSize;

            _replayBuffer = new Queue<ReplayBufferElem>();
            _replayBufferCapacity = p_capacity;

            CreateQTNetwork();
        }

        public void Dispose()
        {
            _optimizer.Dispose();
            _QTnetwork.Dispose();

            foreach(ReplayBufferElem e in _replayBuffer)
            {
                Vector.Release(e.S0);
                Vector.Release(e.S1);
            }
        }

        public float Train(Vector p_state0, int p_action0, Vector p_state1, float p_reward, bool p_final = false)
        {
            float mse = 0;
           
            if (_replayBuffer.Count == _replayBufferCapacity)
            {
                ReplayBufferElem e = _replayBuffer.Dequeue();
                Vector.Release(e.S0);
                Vector.Release(e.S1);
            }

            _replayBuffer.Enqueue(new ReplayBufferElem { S0 = Vector.Copy(p_state0), S1 = Vector.Copy(p_state1), Action = p_action0, Reward = p_reward, Final = p_final });

            if (_replayBuffer.Count >= _batchSize && _batchIndex == _batchSize)
            {
                _batchIndex = 0;
                _sample = _replayBuffer.OrderBy(x => _rnd.Next()).Take(_batchSize);

                int i = 0;

                Stopwatch watch = Stopwatch.StartNew();

                foreach (ReplayBufferElem b in _sample)
                {
                    Vector t = CalcTarget(b.S0, b.S1, b.Action, b.Reward, b.Final);
                    _optimizer.Train(b.S0, t);
                    Vector.Release(t);
                    i++;
                }

                watch.Stop();
#if !DEBUG
                Console.WriteLine("Training time [ms] " + watch.ElapsedMilliseconds);
#endif
            }

            _batchIndex++;

            if (_qtUpdateIndex == _qtUpdateSize)
            {
                _qtUpdateIndex = 0;
                _QTnetwork.OverrideParams(_Qnetwork);
            }

            _qtUpdateIndex++;

            return mse;
        }

        virtual protected Vector CalcTarget(Vector p_s0, Vector p_s1, int p_action, float p_reward, bool p_final)
        {
            float maxQs1a = CalcMaxQa(p_s1);

            // updating phase for Q(s,a)
            _Qnetwork.Activate(p_s0);

            Vector target = Vector.Copy(_Qnetwork.Output);

            if (p_final)
            {
                target[p_action] = p_reward;
            }
            else
            {
                target[p_action] = p_reward + _gamma * maxQs1a;
            }

            return target;

        }

        virtual protected float CalcMaxQa(Vector p_state)
        {
            _QTnetwork.Activate(p_state);

            float maxQa = _QTnetwork.Output[0];

            for (int i = 0; i < _QTnetwork.Output.Size; i++)
            {
                if (_QTnetwork.Output[i] > maxQa)
                {
                    maxQa = _QTnetwork.Output[i];
                }
            }

            return maxQa;
        }

        private void CreateQTNetwork()
        {
            Console.WriteLine("Initializing QTNetwork started");
            _QTnetwork = IOUtils.LoadNetwork(IOUtils.SaveNetwork(_Qnetwork));
            Console.WriteLine("Initializing QTNetwork finished");
        }

        public void SetAlpha(float p_alpha)
        {
            if (_optimizer != null)
            {
                _optimizer.Alpha = p_alpha;
            }
        }

        public Vector Activate(Vector p_input)
        {
            return _Qnetwork.Activate(p_input);
        }

        public Vector Output
        {
            get { return _Qnetwork.Output; }
        }

        public NeuralNetwork Network
        {
            get { return _Qnetwork; }
        }

        public Optimizer Optimizer
        {
            get { return _optimizer; }
        }
    }
}