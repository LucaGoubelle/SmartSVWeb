<?php

    //author: Luca Goubelle

    class Legend
    {
        private $values;

        public function __construct($v, $matter="yes")
        {
            if($matter=="no")
            {
                for($i=0;$i<count($v);$i++)
                {
                    $this->values[] = $i+1;
                }
            }
            else
            {
                $this->values = $v;
            }
        }

        public function getValues()
        {
            return $this->values;
        }

        public function getValueByIndex($index)
        {
            return $this->values[$index];
        }

        public function getCountOfValues()
        {
            return count($this->values);
        }
    }

?>