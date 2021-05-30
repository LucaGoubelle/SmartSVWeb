<?php

	//require_once "normalisationManager.php";

?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>smart village predict | normalise file</title>
	<link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
</head>
<body>
<form action="../../src/quit.php" method="post"><input class="btn btn-danger" name="quitButton" value="Quit" type="submit"></form>
	<div style="text-align: center;">
		<h1>Upload a time serie</h1>
		<form action="../../src/extractCSVData_dataPretreatment.php" enctype="multipart/form-data" method="POST" style="text-align: center;">
			<label class="custom-file-label" for="notNormalisedUploadButton">Choose a time serie to treat</label><br>
			<input class="custom-file-input" type="file" enctype="multipart/form-data" id="notNormalisedUploadButton" name="notNormalisedFile"><br><br>
			<input type="submit" name="uploadButton" class="btn btn-success" value="Ouvrir CSV">
		</form>
	</div>
</body>
</html>