function showColumnIndex(lstCl){
    var selector = document.getElementById("columnName");
    var cnt = lstCl.length;

    for(let i=0;i<cnt;i++){
        selector.innerHTML += "<option value=\""+lstCl[i]+"\">"+lstCl[i]+"</option>";
    }
}

function showColumnsNotNan(lstCl){
    var div = document.getElementById("columnsNotNan");
    var cnt = lstCl.length;

    for(let i=0;i<cnt;i++){
        div.innerHTML += "<label for='colCheckbox_"+String(i+1)+"' >"+lstCl[i]+"</label><input type='checkbox' id='colCheckbox_"+String(i+1)+"' name='columns[]' value='"+lstCl[i]+"' >";
    }
}

function addColumnToListBox(element){
	var box = document.getElementById("selectedMethods");
	if(element.checked==true)
		box.innerHTML += element.id+"\n";
}

function removeColumnToListBox(element){
	var box = document.getElementById("selectedMethods");
	const elementToPop = element.id+'\n';
	if(element.checked==false)
	{
		var newContent = box.innerHTML.replace(elementToPop, '');
		box.innerHTML = newContent;
	}
}

function checkboxChecking(element){
	addColumnToListBox(element);
	removeColumnFromListBox(element);
}

function addAllColumns(arrMth){
	for(let i=0;i<arrMth.length;i++)
		arrMth[i].checked = true;
	for(let i=0;i<arrMth.length;i++)
		addMethodsToListBox(arrMth[i]);

}

function removeAllColumns(arrMth){
	for(let i=0;i<arrMth.length;i++)
		arrMth[i].checked = false;/*
	for(let i=0;i<arrMth.length;i++)
		removeMethodsToListBox(arrMth[i]);*/
	var box = document.getElementById("selectedMethods");
	box.innerHTML = "";
}