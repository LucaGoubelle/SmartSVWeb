<?php

	//author: Luca Goubelle

	require_once "controllers/dataPretreatmentService/normalisationManager.php";
	session_start();
	
	if(isset($_SESSION['normalised_fileName']))
	{
		$fileDoneName = $_SESSION['normalised_fileName'];
		if(isset($_POST['downloadButton']))
		{
			DataPretreatmentManager::downloadPretreatedFile($fileDoneName);
			header('Location: ../vues/normaliseTimeSerie.php');
		}
	}
	else
	{
		?><script>alert("ERR: No file were selected to be normalised !");</script><?php
		header('Location: ../vues/normaliseTimeSerie.php');
	}

?>