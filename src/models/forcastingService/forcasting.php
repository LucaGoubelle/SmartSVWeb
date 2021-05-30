<?php

    //author: Luca Goubelle

    class Forcasting
    {
        private $values;

        public function __construct($f)
        {
            $this->values = $f;
        }

        public function getValues()
        {
            return $this->values;
        }

        public function adaptValuesForGraph($ts)
        {
            for($i=0;$i<count($ts->getValues());$i++)
            {
                array_unshift($this->values, NULL);
            }
        }
    }

?>