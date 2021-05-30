<?php

    //author: Luca Goubelle

    class TimeSerieChart
    {
        private $legend; //contain the legend of the graph
        private $timeSerie = array(); //contain a time serie object array

        public function __construct($_l, $_ts)
        {
            $this->legend = $_l;
            if(is_object($_ts)){ $this->timeSerie[] = $_ts; }
            else if(is_array($_ts))
            { 
                foreach($_ts as $obj)
                {
                    if(is_object($obj) AND $obj instanceof TimeSerie){ $this->timeSerie[] = $obj; }
                }
            }
        }

        public function getLegend()
        {
            return $this->legend->getValue();
        }

        public function getTimeSerie($index)
        {
            return $this->timeSerie[$index]->getValues();
        }

        public function getTimeSerieCount()
        {
            return count($this->timeSerie);
        }
    }

    #-----------------------------------------------------------------

    class MultiTimeSerieChart
    {
        private $legend; //contain the legend of the graph
        private $timeSerie = array(); //contain a time serie object array

        public function __construct($_l, $_ts)
        {
            $this->legend = $_l;
            if(is_object($_ts)){ $this->timeSerie[] = $_ts; }
            else if(is_array($_ts))
            { 
                foreach($_ts as $obj)
                {
                    if(is_object($obj) AND $obj instanceof TimeSerie){ $this->timeSerie[] = $obj; }
                }
            }
        }

        public function getLegend()
        {
            return $this->legend->getValues();
        }

        public function getTimeSerie($index)
        {
            return $this->timeSerie[$index]->getValues();
        }

        public function getTimeSerieCount()
        {
            return count($this->timeSerie);
        }
    }

    #---------------------------------------------------------------------------------------------

    class SingleTimeSerieChart
	{
		private $legend;
		private $timeSerie;

		public function __construct($_l,$_t)
		{
			$this->legend = $_l;
			$this->timeSerie = $_t;
		}

		public function getLegend()
		{
			return $this->legend->getValues();
		}

		public function getTimeSerie()
		{
			return $this->timeSerie->getValues();
		}
	}
?>