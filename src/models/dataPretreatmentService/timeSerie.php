<?php

    //author: Luca Goubelle

    class TimeSerie
    {
        private $values;

        public function __construct($v)
        {
            $this->values = $v;
        }

        public function getValues()
        {
            return $this->values;
        }

        public function getValuesInterval($min,$max)
        {
            $arr = array();
            for($i=$min;$i<=$max;$i++)
            {
                $arr[] = $this->values[$i];
            }
            return $arr;
        }

        public function getMaxOfValues()
        {
            return max($this->values);
        }

        public function getMinOfValues()
        {
            return min($this->values);
        }

        public function getMeanOfValues()
        {
            return array_sum($this->values)/count($this->values);
        }

        public function getCountOfValue()
        {
            return count($this->values);
        }

        public function isNanValuesInTS()
        {
            $isNan = false;
            foreach($this->values as $val)
            {
                if($val=="")
                {
                    $isNan = true;
                    break;
                }
            }
            return $isNan;
        }

        public function isOutliersInTS()
        {
            $isOutliers = false;
            foreach($this->values as $val)
            {
                if(isOutlier($val))
                {
                    $isOutliers = true;
                    break;
                }
            }
            return $isOutliers;
        } 

        public function isNumericColumn()
        {
            $isNumeric = true;
            foreach($this->values as $val)
            {
                if(!is_numeric($val))
                {
                    $isNumeric = false;
                    break;
                }
            }
            return $isNumeric;
        }

        public function getNanValuesCount()
        {
            $cnt = 0;
            foreach($this->values as $val)
            {
                if($val=="")
                {
                    $cnt++;
                }
            }
            return $cnt;
        }

        public function setValueByInterval($vals,$min,$max)
        {
            if(($max-$min)+1 != count($vals))
            {
                echo "ERR: interval cardinal must be length of vals";
            }
            $indexVals = 0;
            for($i=$min;$i<$max+1;$i++)
            {
                $this->values[$i] = $vals[$indexVals];
                $indexVals++;
            }
        }

        public function setValueFromPointAndBeyond($vals, $startpoint)
        {
            $numberOfValueToAddOrPop = (count($this->values)-($startpoint+1))-count($vals);
            if($numberOfValueToAddOrPop<0)
            { 
                for($i=0;$i<abs($numberOfValueToAddOrPop);$i++)
                {
                    $this->values[] = NULL;
                } 
            }
            else if($numberOfValueToAddOrPop>0)
            {
                for($i=0;$i<abs($numberOfValueToAddOrPop);$i++)
                {
                    array_pop($this->values);
                }
            }
            $indexVals = 0;
            for($i=$startpoint;$i<count($this->values);$i++)
            {
                $this->values[$i] = $vals[$indexVals];
                $indexVals++;
            }
        }

        public function adaptValueswithTSFGraph($pred)
        {
            for($i=0;$i<count($pred);$i++)
            {
                array_push($this->values, NULL);
            }
        }
    }

    class TimeSerieCollection
    {
        private $index = array();
        private $ts = array();

        public function __construct($ind, $arrTS)
        {
            if(is_array($ind))
            {
                $this->index = $ind;
            }
            if(is_array($arrTS))
            {
                $this->ts = $arrTS;
            }
            /*else if($arrTS instanceof TimeSerie)
            {
                $this->ts[] = $arrTS;
            }*/
        }

        public function getAllTS()
        {
            return $this->ts;
        }

        public function getIndex()
        {
            return $this->index;
        }

        public function getIndexByIndex($i)
        {
            return $this->index[$i];
        }

        public function getTS($index)
        {
            return $this->ts[$index];
        }

        public function getTimeSerieCount()
        {
            return count($this->ts);
        }

        public function getNumberOfTSValues()
        {
            return count($this->getTS(0)->getValues());
        }

        public function getNanValuesCount()
        {
            $nanCount = 0;
            foreach($this->ts as $ts)
            {
                $nanCount += $ts->getNanValuesCount();
            }
            return $nanCount;
        }

    }

?>