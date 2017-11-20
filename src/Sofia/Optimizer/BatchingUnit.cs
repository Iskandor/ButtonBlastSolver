namespace Sofia
{
    public class BatchingUnit
    {
        private int _batchSize;
        private int _batchIndex;
        private bool _active;
        private bool _batchFinished;

        public BatchingUnit()
        {
            _batchFinished = true;
        }

        public void Init(int p_size = 1)
        {
            _batchSize = p_size;
            _active = p_size > 1;
        }

        public void Reset()
        {
            _batchIndex = 0;
        }

        public void Update()
        {
            if (_active) { 
                _batchFinished = false;
                _batchIndex++;
                if (_batchIndex == _batchSize)
                {
                    _batchFinished = true;
                    Reset();
                }
            }
        }

        public int BatchSize
        {
            get { return _batchSize; }
        }

        public bool IsActive
        {
            get { return _active; }
        }

        public bool IsBatchFinished
        {
            get { return _batchFinished; }
        }
    }
}
