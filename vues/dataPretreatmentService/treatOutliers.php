<?php
	session_start();
	$lstColNotNan = isset($_SESSION['lstColNotNan']) ? $_SESSION['lstColNotNan'] : NULL;
	$totalNan = isset($_SESSION['totalNan']) ? $_SESSION['totalNan'] : NULL;
	$cmdOut = isset($_SESSION['cmdOut']) ? $_SESSION['cmdOut'] : NULL;

	//var_dump($lstColNotNan);
	if($totalNan>0){
		?><script>alert("There is still NAN Values to fill");</script><?php
		header('Location: columnsParameters.php');
	}

	//var_dump($cmdOut);
?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Smart SV Web | Treat Outliers</title>
	<link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
</head>
<body>
	<a href="columnsParameters.php"><input style="color: white;" class="btn btn-warning" value=" << Return to columns"></a><br><br><br>
	<form action="../../src/treatOutliers.php" method="POST">
		<label for="columnName">Column Name</label>
		<select class="form-select" name="columnName">
			<?php
                    for($i=0;$i<count($lstColNotNan);$i++)
                    {
                        echo "<option value=\"".$lstColNotNan[$i]."\">".$lstColNotNan[$i]."</option>";
                    }
            ?>
		</select>
		<label for="methodName">Method Name</label>
		<select class="form-select" name="methodName">
			<option value="EllipticEnvelope">EllipticEnvelope</option>
			<option value="IsolationForest">IsolationForest</option>
			<option value="LocalOutlierFactor">LocalOutlierFactor</option>
			<option value="OneClassSVM">OneClassSVM</option>
		</select>
		<input type="submit" class="btn btn-success" value="Treat Outliers" name="treatOutliers">
	</form>

</body>
</html>