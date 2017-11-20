using System;

namespace Sofia
{
    class RandomGenerator
    {
        private static RandomGenerator _instance;
        private Random _random;

        private RandomGenerator()
        {
            _random = new Random();
        }

        public static RandomGenerator getInstance()
        {
            if (_instance == null)
            {
                _instance = new RandomGenerator();
            }

            return _instance;
        }

        public int Rand(int min = 0, int max = 1)
        {
            return _random.Next(min, max + 1);
        }

        public float Rand(float min, float max)
        {
            return (float)_random.NextDouble() * (max - min) + min;
        }

        public float Normal(float p_mean, float p_sigma)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = 1.0 - _random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)

            return (float)(p_mean + p_sigma * randStdNormal); //random normal(mean,stdDev^2)
        }

        public int GetRandom(float[] pool)
        {
            // get universal probability 
            double u = 0;

            for(int i = 0; i < pool.Length; i++)
            {
                u += pool[i];
            }

            // pick a random number between 0 and u
            double r = _random.NextDouble() * u;

            double sum = 0;
            for (int i = 0; i < pool.Length; i++)
            {
                // loop until the random number is less than our cumulative probability
                if (r <= (sum = sum + pool[i]))
                {
                    return i;
                }
            }
            // should never get here
            return -1;
        }

    }
}
