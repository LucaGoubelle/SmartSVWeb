<?php

    //author: Luca Goubelle

    class MultiTimeSerieForcastedChart
    {
        private $legend;
        private $originalSerie = array();
        private $forcastedSerie = array();

        public function __construct($_l, $_ots, $_fts)
        {
            $this->legend = $_l;
            if(is_object($_fts)){ $this->forcastedSerie[] = $_fts; }
            else if(is_array($_fts))
            {
                foreach($_fts as $obj)
                {
                    if(is_object($obj) AND $obj instanceof TimeSerie){ $this->forcastedSerie[] = $obj; }
                }
            }
            if(is_object($_ots)){ $this->originalSerie[] = $_ots; }
            else if(is_array($_ots))
            {
                foreach($_ots as $obj)
                {
                    if(is_object($obj) AND $obj instanceof TimeSerie){ $this->originalSerie[] = $obj; }
                }
            }
        }

        public function getLegend()
        {
            return $this->legend->getValues();
        }

        public function getOriginalSerie($index)
        {
            return $this->originalSerie[$index]->getValues();
        }

        public function getForcastedSerie($index)
        {
            return $this->forcastedSerie[$index]->getValues();
        }

        public function getOriginalSerieCount()
        {
            return count($this->originalSerie);
        }

        public function getForcastedSerieCount()
        {
            return count($this->forcastedSerie);
        }
    }

    #----------------------------------------------------------------------------------

    class SingleTimeSerieForcastedChart
    {
        private $legend;
        private $timeSerieOriginal;
        private $forcastedSerie;

        public function __construct($_l, $_tso, $_fts)
        {
            $this->legend = $_l;
            $this->timeSerieOriginal = $_tso;
            $this->forcastedSerie = $fts;
        }

        public function getLegend()
        {
            return $this->legend->getValues();
        }

        public function getOriginalSerie()
        {
            return $this->timeSerieOriginal->getValues();
        }

        public function getForcastedSerie()
        {
            return $this->forcastedSerie->getValues();
        }
    }

?>