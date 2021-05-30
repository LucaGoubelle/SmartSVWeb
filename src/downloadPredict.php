<?php
	//define("PATH_ROOT", dirname(__FILE__));

	require "controllers/forcastingService/forcastingManager.php";
	session_start();

	$nrmF = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;

	if(isset($_POST['downloadP'])){
		ForcastingManager::downloadFile("data/forcasted/".$nrmF);
		ForcastingManager::clearFolders($nrmF);
		session_unset();
		header('Location: ../vues/forcastingService/uploadTimeSerie.php');
	}

?>