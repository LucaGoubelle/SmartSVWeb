<?php

	require "src/controllers/dataPretreatmentService/csvTSDataManager.php";

	$csvArr = CSV_TS_Data_Manager::arrayFromCSV("src/data/__matrix_form_407pl.csv");
	var_dump($csvArr);

?>