<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Smart SV prediction | Upload file of time serie</title>
	<link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
</head>
<body>
<form action="../../src/quit.php" method="post"><input class="btn btn-danger" name="quitButton" value="Quit" type="submit"></form>
	<div style="text-align: center;">
		<h1>Upload a time serie</h1>
		<form action="../../src/extractCSVData_forcasting.php" enctype="multipart/form-data" method="POST" style="text-align: center;">
			<label class="custom-file-label" for="timeSerieUploadButton">Choose a time serie already treated</label><br>
			<input class="custom-file-input" type="file" enctype="multipart/form-data" id="timeSerieUploadButton" name="timeSerieFile"><br><br>
			<input type="submit" name="uploadButton" class="btn btn-success" value="Afficher">
		</form>
	</div>
</body>
</html>