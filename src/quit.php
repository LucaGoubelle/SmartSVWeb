<?php

	require "controllers/forcastingService/forcastingManager.php";
	require "controllers/dataPretreatmentService/dataPretreatmentManager.php";
    session_start();

    if(isset($_POST["quitButton"])){
    	DataPretreatmentManager::clearFolders($nrmF);
    	ForcastingManager::clearFolders($nrmF);
        session_unset();
        header('Location: ../index.php');
    }

?>