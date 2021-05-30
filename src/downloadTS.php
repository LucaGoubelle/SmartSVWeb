<?php
	ob_start();

	require "controllers/dataPretreatmentService/dataPretreatmentManager.php";
	session_start();

	$nrmF = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;

	if(isset($_POST['downloadTS'])){
		DataPretreatmentManager::downloadFile("data/saved/".$nrmF);
		DataPretreatmentManager::clearFolders($nrmF);
		session_unset();
		header('Location: ../vues/dataPretreatmentService/uploadTimeSerieFile.php');
	}
	else if(isset($_POST['dfNormalise01'])){
		$cmdOut = DataPretreatmentManager::normaliseDF($nrmF, $_POST["columnIndex"]);
		copy("data/normalised01/".$nrmF, "data/saved/".$nrmF);
		$_SESSION["cmdOut"] = $cmdOut;
		?><script>alert("Normalisation done !");</script><?php
		header('Location: ../vues/dataPretreatmentService/columnsParameters.php');
	}

	ob_end_flush();
?>